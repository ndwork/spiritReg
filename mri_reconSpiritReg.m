
function img = mri_reconSpiritReg( kData, varargin )
  % img = mri_reconSpiritReg( kData [, 'cs', true/false, 'gamma', gamma, 
  %       'noiseVar', noiseVar, 'sACR', sACR, 'sMaps', sMaps, 'support', support, 
  %       'verbose', verbose, 'wSize', wSize ] )
  %
  % The Data consistency term, by default is Adc = M F S
  % If a support is supplied, but not a noise bound, then Adc = M F S PT
  %
  % The problem to solve, without any other parameters, is
  % minimize || Adc x - b ||_2
  %
  % When compressed sensing is set, the problem becomes
  %   minimize || Psi y ||_{2,1} subject to || M F y(c) - b(c) ||_2^2 / numel(b(c)) <= noiseVar
  %     where y is an array of size M x N x C representing images of all coils
  %   This is equivalent to
  %     || M F y(c) - b(c) ||_2^2 <= noiseVar * numel( b(c) )
  %   a noiseVar is required when using compressed sensing
  %   by default, Psi is the Daubechies-4 wavelet transform
  %   After y is created, the single image x is reconstructed by solving
  %     minimize || S x - y ||_2 with LSQR
  %
  % If wSize and sACR are supplied, then the problem is also constrained by
  %   || ( W - I ) F S PT x ||_2^2 / N <= gamma
  %   where N is the number of elements in kData
  % OR
  %   || ( W - I ) F S(c) PT x ||_2^2 / nImg <= gamma(c) for all c = 1, 2, ... C
  %   where nImg is the number of pixels in the image and C is the number of coils
  % By default, gamma is determined automatically as a vector, one value per coil
  %
  % If support is supplied and a noise bound, then the problem is also constrained by
  % || P_C x ||_2^2 / ( nImg - nSupport ) <= noiseVar
  %   This is handled similarly to "Parameter-free Parallel Imaging and
  %   Compressed Sensing" by Tamir et al. ISMRM, 2018
  %
  % Inputs:
  % kData - a complex matrix of size M x N x C, where C is the number of coils
  %   Uncollected data will have a value of 0.
  % sACR - a scalar or a 2 element array that specifies the size of the autocalibration region.
  %  If it's a scalar, then the size is assumed to be [ sACR sACR ].
  % wSize - either a scalar or a two element array of odd vlaues that specifies the size of the interpolation 
  %   kernel.  If wSize is a scalar, then the kernel is assumed to have size [ wSize wSize ].
  %
  % Optional Inputs:
  % gamma - either empty, a scalar, or an array of nonnegative values (one per coil)
  %   If nonempty, then this function solves the problem involving gamma
  % noiseVar - positive scalar that bounds the average value of the magnitude of
  %   the noise squared of a single pixel
  % Psi - a sparsifying transformation.  Either a matrix or a function with prototype
  %       out = Psi( in, op ) where op is optional and either 'transp' or 'notransp'
  %
  % Outputs
  % img - a 2D complex array of size M x N that is the output image.
  %
  % Written by Nicholas Dwork, Copyright 2025
  %
  % https://github.com/ndwork/dworkLib.git
  %
  % This software is offered under the GNU General Public License 3.0.  It
  % is offered without any warranty expressed or implied, including the
  % implied warranties of merchantability or fitness for a particular
  % purpose.

% TODO: add in compressed sensing for multiple coils
% TODO: take a different noise bound into account for each coil
% TODO: replace L1 norm in compressed sensing with Huber penalty

  if nargin < 2
    disp([ 'Usage: img = mri_reconSpiritReg( kData, sMaps [, ''cs'', true/false, ' ...
           ' ''gamma'', gamma, ''sACR'', sACR, ''support'', support, ' ...
           ' ''verbose'', verbose, ''wSize'', wSize ] )' ]);
    if nargout > 0, img = []; end
    return
  end

  p = inputParser;
  p.addParameter( 'cs', false );
  p.addParameter( 'doChecks', false );
  p.addParameter( 'gamma', [], @isnonnegative );
  p.addParameter( 'noiseVar', [], @isnonnegative );
  p.addParameter( 'optAlg', [], @(x) true );
  p.addParameter( 'Psi', [] );
  p.addParameter( 'sACR', [], @ispositive );
  p.addParameter( 'sMaps', [], @isnumeric );
  p.addParameter( 'support', [], @(x) isnumeric(x) || islogical(x) );
  p.addParameter( 'verbose', true );
  p.addParameter( 'wSize', [], @ispositive );
  p.parse( varargin{:} );
  cs = p.Results.cs;
  doChecks = p.Results.doChecks;
  gamma = p.Results.gamma;
  noiseVar = p.Results.noiseVar;
  optAlg = p.Results.optAlg;
  Psi = p.Results.Psi;
  sACR = p.Results.sACR;
  sMaps = p.Results.sMaps;
  support = p.Results.support;
  verbose = p.Results.verbose;
  wSize = p.Results.wSize;

  sKData = size( kData );
  nKData = numel( kData );
  sImg = sKData(1:2);
  nCoils = size( kData, 3 );
  nImg = prod( sImg );
  if isscalar( sACR ), sACR = [ sACR sACR ]; end
  if isscalar( wSize ), wSize = [ wSize wSize ]; end

  if numel( wSize ) > 0  &&  numel( sACR ) == 0
    error( 'Must supply sACR if you supply wSize' );
  end
  if numel( wSize ) > 0  &&  nCoils > 1
    error( 'Cannot perform spiritReg with only one coil' );
  end
  if ( cs == true || cs > 0 ) &&  numel( noiseVar ) == 0
    error( 'Must supply a noise bound when doing compressed sensing' );
  end

  if min( mod( wSize, 2 ) ) < 1, error( 'wSize elements must be odd' ); end
  if numel( sMaps ) == 0
    if nCoils > 1, error( 'Must supply sMaps with multiple coils' ); end
    sMaps = ones( sImg );
  end

  sampleMask = ( kData ~= 0 );
  b = kData( sampleMask == 1 );
  nb = numel( b );
  sOut = [ sImg(:)' nCoils ];
  nUnknown = prod( sOut ) - nb;

  img0 = zeros( sImg );
  nSupport = nImg;
  if numel( support ) > 0
    nSupport = sum( support(:) );
    if ( numel( noiseVar ) == 0  ||  noiseVar == 0 )
      img0 = img0( support ~= 0 );
    end
  end
  sImg0 = size( img0 );
  nImg0 = numel( img0 );


  %% Create linear transformations and their adjoints

  function out = applyM( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( sampleMask == 1 );
    else
      out = zeros( [ sImg nCoils ] );
      out( sampleMask == 1 ) = in;
    end
  end

  function out = applyMc( in, op )  % Apply M complement
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( sampleMask == 0 );
    else
      out = zeros( [ sImg nCoils ] );
      out( sampleMask == 0 ) = in;
    end
  end

  function out = applyP( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( support == 1 );
    else
      out = zeros( sImg );
      out( support == 1 ) = in;
    end
  end

  function out = applyF( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = fftshift2( fft2( ifftshift2( in ) ) );
    elseif strcmp( op, 'inv' )
      out = fftshift2( ifft2( ifftshift2( in ) ) );
    elseif strcmp( op, 'transpInv' )
      out = fftshift2( ifft2h( ifftshift2( in ) ) );
    elseif strcmp( op, 'transp' )
      out = fftshift2( fft2h( ifftshift2( in ) ) );
    else
      error( 'unrecognized operation' );
    end
  end

  function out = applyS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = bsxfun( @times, sMaps, in );
    else
      out = sum( conj(sMaps) .* in, 3 );
    end
  end

  function out = apply_vS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyS( reshape( in, sImg ), op );
    else
      out = applyS( reshape( in, sKData ), op );
    end
    out = out(:);
  end

  function out = applyFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      if numel( img0 ) ~= nImg
        in = applyP( in, 'transp' );
      end
      out = applyF( applyS( in ) );
    else
      out = applyS( applyF( in, op ), op );
      if numel( img0 ) ~= nImg
        out = applyP( out, 'notransp' );
      end
    end
  end

  function out = applyMF( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyM( applyF( in ) );
    else
      out = applyF( applyM( in, op ), op );
    end
  end

  function out = apply_vMF( in, op ) %vectorized version of applyMF
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyMF( reshape( in, sKData ) );
    else
      out = applyMF( in, 'transp' );
      out = out(:);
    end
  end

  function out = applyMFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyM( applyFS( in ) );
    else
      out = applyFS( applyM( in, op ), op );
    end
  end

  function out = apply_vMFS( in, op )  % vectorized version of applyMFS
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyMFS( reshape( in, sImg0 ) );
    else
      out = applyMFS( in, 'transp' );
      out = out(:);
    end
  end

  if numel( wSize ) > 0
    acr = cropData( kData, [ sACR(1) sACR(2) nCoils ] );
    if numel( gamma ) == 0
      [w, gamma] = findW( acr, wSize );
    elseif isscalar( gamma )
      gamma = gamma * ones( nCoils, 1 );
    end
    flipW = padData( flipDims( w, 'dims', [1 2] ), [ sImg nCoils nCoils ] );
  end

  function out = applyW( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = squeeze( sum( circConv2( flipW, in ), 3 ) );
    else
      in = repmat( reshape( in, [ sImg 1 nCoils ] ), [ 1 1 nCoils, 1] );
      out = circConv2( flipW, in, 'transp', 'ndimsOut', ndims(kData) );
    end
  end

  function out = applyWmI( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyW( in ) - in;
    else
      out = applyW( in, 'transp' ) - in;
    end
  end

  function out = applyWmIFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      FSin = applyFS( in );
      out = applyW( FSin ) - FSin;
    else
      out = applyFS( applyW( in, 'transp' ) - in, 'transp' );
    end
  end

  function out = concatMFSWmIFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      FSin = applyFS( in );
      MFSin = applyM( FSin );
      WmIFSin = applyWmI( FSin );
      out = [ MFSin(:); WmIFSin(:); ];
    else
      n1 = nb;
      n2 = n1 + nKData;
      MTin1 = reshape( applyM( in( 1 : n1 ), op ), sKData );
      WHmIHin3 = reshape( applyWmI( reshape( in( n1+1 : n2 ), sKData ), op ), sKData );
      out = applyFS( MTin1 + WHmIHin3, op );
    end
  end

  function out = concatMFSIWmIFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      FSin = applyFS( in );
      MFSin = applyM( FSin );
      WmIFSin = applyWmI( FSin );
      out = [ MFSin(:); in(:); WmIFSin(:); ];
    else
      n1 = nb;
      n2 = n1 + nImg0;
      n3 = n2 + nKData;
      MTin1 = reshape( applyM( in( 1 : n1 ), op ), sKData );
      in2 = reshape( in( n1+1 : n2 ), sImg0 );
      WHmIHin3 = reshape( applyWmI( reshape( in( n2+1 : n3 ), sKData ), op ), sKData );
      out = applyFS( MTin1 + WHmIHin3, op ) + in2;
    end
  end

  function out = defaultPsi( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = zeros( sKData );
      for coilIndx = 1 : nCoils
        out(:,:,coilIndx) = wtDaubechies2( in(:,:,coilIndx), wavSplit );
      end
    else
      out = zeros( sKData );
      for coilIndx = 1 : nCoils
        out(:,:,coilIndx) = iwtDaubechies2( in(:,:,coilIndx), wavSplit );
      end
    end
  end

  if numel( Psi ) == 0
    wavSplit = makeWavSplit( sImg );
    Psi = @defaultPsi;
    nPsiOut = numel( Psi( rand( sKData ) ) );
  end

  function out = PsiFinvMcT( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = Psi( applyF( applyMc( in, 'transp' ), 'inv' ) );
    else
      out = applyMc( applyF( Psi( in, 'transp' ), 'transpInv' ) );
    end
  end

  function out = concatPsiMF( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      PsiIn = Psi( in );
      MFin = applyMF( in );
      out = [ PsiIn(:); MFin(:); ];
    else
      PsiIn1 = Psi( reshape( in(1:nPsiOut), [ sImg nCoils ] ), op );
      FHMTin2 = applyMF( in(nPsiOut+1:end), op );
      out = PsiIn1 + FHMTin2;
    end
  end

  %% Check the adjoints
doChecks = true
  if doChecks == true
    if nImg0 ~= nImg
      [chkP,errChkP] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyP );
      if chkP == true
        disp( 'Check of P Adjoint passed' );
      else
        error([ 'Check of P Adjoint failed with error ', num2str(errChkP) ]);
      end
    end

    [chkS,errChkS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyS );
    if chkS == true
      disp( 'Check of S Adjoint passed' );
    else
      error([ 'Check of S Adjoint failed with error ', num2str(errChkS) ]);
    end

    [chk_vS,errChk_vS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @apply_vS );
    if chk_vS == true
      disp( 'Check of vS Adjoint passed' );
    else
      error([ 'Check of vS Adjoint failed with error ', num2str(errChk_vS) ]);
    end

    [chkFS,errChkFS] = checkAdjoint( rand( sImg0 ) + 1i * rand( sImg0 ), @applyFS );
    if chkFS == true
      disp( 'Check of FS Adjoint passed' );
    else
      error([ 'Check of FS Adjoint failed with error ', num2str(errChkFS) ]);
    end

    [chkF,errChkF] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyF );
    if chkF == true
      disp( 'Check of F Adjoint passed' );
    else
      error([ 'Check of F Adjoint failed with error ', num2str(errChkF) ]);
    end

    [chkMF,errChkMF] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyMF );
    if chkMF == true
      disp( 'Check of MF Adjoint passed' );
    else
      error([ 'Check of MF Adjoint failed with error ', num2str(errChkMF) ]);
    end

    nKData = prod( sKData );
    [chk_vMF,errChk_vMF] = checkAdjoint( rand( nKData, 1 ) + 1i * rand( nKData, 1 ), @apply_vMF );
    if chk_vMF == true
      disp( 'Check of vMF Adjoint passed' );
    else
      error([ 'Check of vMF Adjoint failed with error ', num2str(errChk_vMF) ]);
    end

    [chkMFS,errChkMFS] = checkAdjoint( rand( sImg0 ) + 1i * rand( sImg0 ), @applyMFS );
    if chkMFS == true
      disp( 'Check of MFS Adjoint passed' );
    else
      error([ 'Check of MFS Adjoint failed with error ', num2str(errChkMFS) ]);
    end

    [chk_vMFS,errChk_vMFS] = checkAdjoint( rand( sImg0 ) + 1i * rand( sImg0 ), @apply_vMFS );
    if chk_vMFS == true
      disp( 'Check of vMFS Adjoint passed' );
    else
      error([ 'Check of vMFS Adjoint failed with error ', num2str(errChk_vMFS) ]);
    end

    [chkPsiFinvMcT,errChkPsiFinvMcT] = checkAdjoint( ...
      rand( nUnknown, 1 ) + 1i * rand( nUnknown, 1 ), @PsiFinvMcT );
    if chkPsiFinvMcT == true
      disp( 'Check of PsiFinvMcT Adjoint passed' );
    else
      error([ 'Check of PsiFinvMcT Adjoint failed with error ', num2str(errChkPsiFinvMcT) ]);
    end

    if numel( wSize ) > 0
      [chkW,errChkW] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyW );
      if chkW == true
        disp( 'Check of W Adjoint passed' );
      else
        error([ 'Check of W Adjoint failed with error ', num2str(errChkW) ]);
      end

      [chkWmI,errChkWmI] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyWmI );
      if chkWmI == true
        disp( 'Check of W minus I Adjoint passed' );
      else
        error([ 'Check of W minus I Adjoint failed with error ', num2str(errChkWmI) ]);
      end

      [chkWmIFS,errChkWmIFS] = checkAdjoint( rand( sImg0 ) + 1i * rand( sImg0 ), @applyWmIFS );
      if chkWmIFS == true
        disp( 'Check of WmIFS Adjoint passed' );
      else
        error([ 'Check of WmIFS Adjoint failed with error ', num2str(errChkWmIFS) ]);
      end

      [chkMFSIWmIFS,errChkMFSIwWmIFS] = checkAdjoint( ...
        rand( sImg0 ) + 1i * rand( sImg0 ), @concatMFSIWmIFS );
      if chkMFSIWmIFS == true
        disp( 'Check of concatMFSIWmIFS Adjoint passed' );
      else
        error([ 'Check of concatMFSIWmIFS Adjoint failed with error ', num2str(errChkMFSIwWmIFS) ]);
      end
    end

    [chkPsi,errChkPsi] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), Psi );
    if chkPsi == true
      disp( 'Check of Psi Adjoint passed' );
    else
      error([ 'Check of Psi Adjoint failed with error ', num2str(errChkPsi) ]);
    end

    [chk_concatPsiMF,errChk_concatPsiMF] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), ...
      @concatPsiMF );
    if chk_concatPsiMF == true
      disp( 'Check of concatPsiMF Adjoint passed' );
    else
      error([ 'Check of concatPsiMF Adjoint failed with error ', num2str(errChk_concatPsiMF) ]);
    end
    
  end


  %% Create the functions for the optimization algorithms

  nInDC = nb;
  gDC = @( x ) 0.5 * norm( x - b )^2;

  prox_gDCH = @(in,t) proxL2Sq( in, t, b );

  nbc = nb / nCoils;
  function out = gDC_ind( in )
    diffs = reshape( in - b, [], nCoils );
    if max( LpNorms( diffs, 2, 1 ).^2 ) > nbc * noiseVar
      out = Inf;
    else
      out = 0;
    end
  end

  radGdcInd = sqrt( nbc * noiseVar );
  bGdcInd = reshape( b, [], nCoils );
  function out = prox_gDC_ind( in, t )   %#ok<INUSD>
    in = reshape( in, [], nCoils );
    diffs = in - reshape( b, [], nCoils );
    normsSq = LpNorms( diffs, 2, 1 );
    out = in;
    for c = 1 : nCoils
      if normsSq(c) > radGdcInd
        out(:,c) = projectOntoBall( in(:,c) - bGdcInd(:,c), radGdcInd ) + bGdcInd(:,c);
      end
    end
  end

  if nCoils == 1
    % || x - b ||^2 / nb <= noiseVar
    % gDC( x ) = 0.5 || x - b ||^2 <= 0.5 * noiseVar * nb
    % || x - b ||_2 <= sqrt( noiseVar * nb )
    if numel( noiseVar ) > 0, bound_gDC_ind = 0.5 * noiseVar * nb; end
    gDC_indH = @( x ) indicatorFunction( gDC( x ), [ 0 bound_gDC_ind ] );

    if numel( noiseVar ) > 0  &&  noiseVar == 0
      prox_gDC_indH = @(in,t) b;
    else
      rad_gDH_indH = sqrt( noiseVar * nb );
      prox_gDC_indH = @(in,t) projectOntoBall( in - b, rad_gDH_indH, 2 ) + b;
    end
  else
    gDC_indH = @gDC_ind;
    prox_gDC_indH = @prox_gDC_ind;
  end

  if numel( support ) > 0  &&  numel( noiseVar ) > 0  &&  noiseVar > 0
    nOutSupport = ( nImg - nSupport );
    rSupportNoiseBall = sqrt( nOutSupport * noiseVar );  
  end

  nInSupport = nImg0;
  function out = gSupport( x )
    outsideSupport = x( support == 0 );
    out = indicatorFunction( norm( outsideSupport, 'fro' )^2 / nOutSupport, [0 noiseVar] );
  end

  function out = prox_gSupport( x, t )   %#ok<INUSD>
    outsideSupport = x( support == 0 );
    out = x;
    out( support == 0 ) = projectOntoBall( outsideSupport, rSupportNoiseBall, 2 );
  end

  gSupportH = [];
  prox_gSupportH = [];
  if numel( support ) > 0  &&  numel( noiseVar ) > 0  &&  noiseVar > 0
    gSupportH = @gSupport;
    prox_gSupportH = @prox_gSupport;
  end

  nInSpiritReg = nKData;
  function out = gSpiritReg( x )
    x = reshape( x, sKData );
    for coilIndx = 1 : nCoils
      if norm( x(:,:,coilIndx), 'fro' )^2 / nImg > gamma(coilIndx)
        out = Inf;
        return
      end
    end
    out = 0;
  end

  function out = prox_gSpiritReg( x, t )   %#ok<INUSD>
    % indicator( || x ||_2^2 / nImg <= gamma(coilIndx) ) is equivalent to
    % indicator( || x(:,:,coilIndx) ||_Fro <= sqrt( gamma( coilIndx ) * nImg ) )
    x = reshape( x, sKData );
    out = zeros( sKData );
    for coilIndx = 1 : nCoils
      normCoilX = norm( x(:,:,coilIndx), 'fro' );
      ballRadius = sqrt( gamma( coilIndx ) * nImg );
      if normCoilX <= ballRadius
        out(:,:,coilIndx) = x(:,:,coilIndx);
      else
        out(:,:,coilIndx) = ( ballRadius / normCoilX ) * x(:,:,coilIndx);
      end
    end
    out = out(:);
  end

  if numel( gamma ) > 0
    gSpiritRegH = @gSpiritReg;
    prox_gSpiritRegH = @prox_gSpiritReg;
  else
    gSpiritRegH = [];
    prox_gSpiritRegH = [];
  end

  function out = prox_gCS_multiCoil( in, t )
    out = proxL2L1( reshape( in, [ sImg, nCoils ] ), t );
    out = out(:);
  end

  if nCoils == 1
    if noiseVar == 0
      bCS = Psi( applyF( kData, 'inv' ) );
      gCS = @(in) norm( in(:) + bCS(:), 1 );
      prox_gCS = @(in,t) proxL1Complex( in + bCS, t ) - bCS;
    else
      gCS = @(in) norm( in(:), 1 );
      prox_gCS = @(in,t) proxL1Complex( in, t );
    end
  else
    if noiseVar == 0
      error( 'Not yet implemented' );
    else
      gCS = @(in) normL2L1( in );
      prox_gCS = @prox_gCS_multiCoil;
    end
  end

  % Create f, g, and A so that we are optimizing f(x) + g(Ax)
  f = [];  proxf = [];
  g = [];  proxg = [];
  applyA = [];
  usePDHG = false;
  useFISTA = false;

  if nCoils == 1
    img0 = fftshift2( ifft2( ifftshift2( kData ) ) );
  else

    if cs == true
      img0 = fftshift2( ifft2( ifftshift2( kData ) ) );

    else
      tol = 1d-6;
      nMaxIter = 100;
      [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( ...
        @apply_vMFS, b, tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>
      img0 = reshape( img, sImg );
    end
  end

  if cs == true
    usePDHG = true;

    if numel( gSupportH ) == 0  &&  numel( gSpiritRegH ) == 0

      if noiseVar == 0
        % minimize || Psi F^{-1} ( Mc^T kUnknown + kData ) ||_1
        % minimize || Psi F^{-1} Mc^T kUnknown + Psi F^{-1} kData ) ||_1
        applyA = @(in,op) PsiFinvMcT( in, op );
        g = gCS;  % g(in) = || in + Psi F^{-1} kData ||_1
        proxg = prox_gCS;
        x0 = zeros( nUnknown, 1 );

      elseif noiseVar > 0
TODO: since works when noiseVar == 0, problem looks to be in the indicator function or its prox
        applyA = @(in,op) concatPsiMF( in, op );
        n1 = nPsiOut;
        n2 = n1 + nInDC;
        g = @(in) gCS( in( 1 : n1 ) ) + ...
                  gDC_indH( in( n1 + 1 : n2 ) );
        proxg = @(in,t) [ prox_gCS( in( 1 : n1 ), t ); ...
                          prox_gDC_indH( in( n1 + 1 : n2 ), t ); ];
      end

    else
      error( 'Not yet implemented' );
    end
  
  else

    if numel( gSupportH ) > 0  ||  numel( gSpiritRegH ) > 0
  
      if numel( gSpiritRegH ) > 0 && numel( gSupportH ) > 0
        usePDHG = true;
        applyA = @concatMFSIWmIFS;
        n1 = nInDC;
        n2 = n1 + nInSupport;
        n3 = n2 + nInSpiritReg;
        g = @(in) gDC( in( 1 : n1 ) ) + ...
                  gSupportH( in( n1 + 1 : n2 ) ) + ...
                  gSpiritRegH( in( n2 + 1 : n3 ) );
        proxg = @(in,t) [ prox_gDCH( in( 1 : n1 ), t ); ...
                          prox_gSupportH( in( n1 + 1 : n2 ), t ); ...
                          prox_gSpiritRegH( in( n2 + 1 : n3 ), t ); ];
  
      elseif numel( gSpiritRegH ) > 0 && numel( gSupportH ) == 0
        usePDHG = true;
        applyA = @(in,op) concatMFSWmIFS( in, op );
        n1 = nInDC;
        n2 = n1 + nInSpiritReg;
        g = @(in) gDC( in( 1 : n1 ) ) + ...
                  gSpiritRegH( in ( n1 + 1 : n2 ) );
        proxg = @(in,t) [ prox_gDCH( in( 1 : n1 ), t ); ...
                          prox_gSpiritRegH( in( n1 + 1 : n2 ), t ); ];
  
      elseif numel( gSpiritRegH ) == 0  &&  numel( gSupportH ) > 0
        useFISTA = true;
        f = @(in) 0.5 * norm( applyMFS( in ) - b, 'fro' )^2;
        ATb = applyMFS( b, 'transp' );
        fGrad = @(in) applyMFS( applyMFS( in ), 'transp' ) - ATb;
        g = gSupportH;
        proxg = prox_gSupportH;
  
      end
    end
  end % if cs == true


  %% Perform the optimization

  if usePDHG == true
    proxgConj = @(in,s) proxConj( proxg, in, s );

    normA = powerIteration( applyA, rand(size(x0)) + 1i * rand( size(x0) ) );
    tau = 1 / normA;

    if numel( optAlg ) == 0, optAlg = 'pdhgWLS'; end

    if strcmp( optAlg, 'pdhg' )
      [y, objValues, relDiffs] = pdhg( x0, proxf, proxgConj, tau, 'A', applyA, 'f', f, 'g', g, ...
        'N', 1000, 'normA', normA, 'printEvery', 10, 'verbose', verbose );   %#ok<ASGLU>
    elseif strcmp( optAlg, 'pdhgWLS' )
      [y, objValues] = pdhgWLS( x0, proxf, proxgConj, 'A', applyA, 'tau', tau, ...
        'f', f, 'g', g, 'N', 100, 'verbose', verbose );   %#ok<ASGLU>
    else
      error( 'Unrecognized optimization algorithm' );
    end

  elseif useFISTA == true
    ATA = @(in) applyMFS( applyMFS( in ), 'transp' );
    L = powerIteration( ATA, rand( sImg ), 'symmetric', true );
    t0 = 1 / L;
    if verbose == true
      [y,objValues,relDiffs] = fista_wLS( img0, f, fGrad, proxg, 'h', g, 't0', t0, ...
        'verbose', verbose );   %#ok<ASGLU>
    else
      y = fista_wLS( img0, f, fGrad, proxg, 'h', g, 'verbose', verbose );
    end

  else  % use LSQR
    tol = 1d-6;
    nMaxIter = 100;
    [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( ...
      @apply_vMFS, b, tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>
  end


  %% Post processing to reconstruct the image after solving the problem

  if cs == true

    if noiseVar == 0
      img = applyF( applyMc( y, 'transp' ) + kData, 'inv' );
    end

    if nCoils > 1
      tol = 1d-6;
      nMaxIter = 100;
      img0 = mri_reconRoemer( img );
      [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( ...
        @apply_vS, img(:), tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>
  
      img = reshape( img, sImg );
    end

  else
    if nImg0 ~= nImg
      img = applyP( y, 'transp' );
    else
      img = reshape( y, sImg );
    end

  end

end


%% SUPPORT FUNCTIONS

function [w, gammas] = findW( acr, wSize )
  %-- Find the interpolation coefficients w
  sACR = size( acr );
  nCoils = sACR( 3 );

  w = cell( 1, 1, 1, nCoils );
  gammas = zeros( nCoils, 1 );
  parfor coilIndx = 1 : nCoils
    A = zeros( (  sACR(2) - wSize(2) + 1 ) * ( sACR(1) - wSize(1) + 1 ), ...
                  wSize(1) * wSize(2) * nCoils - 1 );   %#ok<PFBNS>
    if size( A, 1 ) < size( A, 2 ), error( 'The size of the ACR is too small for this size kernel' ); end
    b = zeros( size(A,1), 1 );
    pt2RemoveIndx = ceil( wSize(1)/2 ) + floor( wSize(2)/2 ) * wSize(1) + ( coilIndx - 1 ) * wSize(2) * wSize(1);
    aIndx = 1;
    for i = ceil( wSize(2)/2 ) : sACR(2) - floor( wSize(2)/2 )
      for j = ceil( wSize(1)/2 ) : sACR(1) - floor( wSize(1)/2 )
        subACR = acr( j - floor( wSize(2)/2 ) : j + floor( wSize(2)/2 ), ...
                               i - floor( wSize(2)/2 ) : i + floor( wSize(2)/2 ), : );   %#ok<PFBNS>
        subACR = subACR(:);
        b( aIndx ) = subACR( pt2RemoveIndx );
        subACR = [ subACR( 1 : pt2RemoveIndx-1 ); subACR( pt2RemoveIndx + 1 : end ); ];
        A( aIndx, : ) = transpose( subACR );
        aIndx = aIndx + 1;
      end
    end
    wCoil = A \ b(:);
    gammas( coilIndx ) = norm( A * wCoil - b )^2 / numel(b);
    wCoil = [ wCoil( 1 : pt2RemoveIndx - 1 ); 0; wCoil( pt2RemoveIndx : end ); ];
    w{1,1,1,coilIndx} = reshape( wCoil, [ wSize nCoils ] );
  end
  w = cell2mat( w );
  %gammas = mean( gammas ) / nCoils;
end
