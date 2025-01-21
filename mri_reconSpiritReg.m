
function img = mri_reconSpiritReg( kData, sACR, wSize, sMaps, varargin )
  % img = mri_reconSpiritReg( kData, sACR, wSize, sMaps [, 'gamma', gamma, 
  %   'support', support, 'lambda', lambda ] )
  %
  % minimize || M F S PT x - b ||_2^2 + lambda || ( W - I ) F S x ||_2^2
  % ... equivalently ...
  % minimize ||  [             M            ] F S x - [ b ] ||
  %          ||  [ sqrt( lambda ) ( W - I ) ]         [ 0 ] ||_2
  %
  % OR
  %
  % minimize (1/2) || M F S PT x - b ||_2^2
  % subject to || ( W - I ) F S x ||_2^2 / N <= gamma
  %   where N is the number of elements in kData
  %
  % OR
  %
  % minimize (1/2) || M F S PT x - b ||_2^2
  % subject to || ( W - I ) F S(c) x ||_2^2 / nPix <= gamma(c) for all c = 1, 2, ... C
  %   where nPix is the number of pixels in the image
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
  % lambda = either empty or a nonnegative scalar.
  %   If nonempty, then this function solves the problem involving lambda
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

%TODO: replace pdhgWLS with gPDHG_wLS
%TODO: vectorize the loops in applyW
%TODO: add non-rectangular FOV
%TODO: add preconditioning

  if nargin < 3
    disp([ 'Usage: img = mri_reconSpiritReg( kData, sACR, wSize [, ''lambda'', lambda,', ...
           ' ''gamma'', gamma ] )' ]);
    if nargout > 0, img = []; end
    return
  end

  if min( mod( wSize, 2 ) ) < 1, error( 'wSize elements must be odd' ); end
  if isscalar( sACR ), sACR = [ sACR sACR ]; end
  if isscalar( wSize ), wSize = [ wSize wSize ]; end

  p = inputParser;
  p.addParameter( 'doChecks', false );
  p.addParameter( 'gamma', [], @isnonnegative );
  p.addParameter( 'lambda', [], @isnonnegative );
  p.addParameter( 'support', [], @(x) isnumeric(x) || islogical(x) );
  p.parse( varargin{:} );
  doChecks = p.Results.doChecks;
  gamma = p.Results.gamma;
  lambda = p.Results.lambda;
  support = p.Results.support;

  sImg = size( kData, [1 2] );
  nCoils = size( kData, 3 );

  if numel( lambda ) == 0  &&  numel( gamma ) == 0
    doFindGamma = true;
  else
    doFindGamma = false;
  end
  if numel( lambda ) == 0, lambda = 0; end

  sampleMask = ( kData ~= 0 );
  nSamples = nnz( sampleMask );
  if numel( support ) > 0, nSupport = sum( support(:) ); end

  function out = applyM( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( sampleMask == 1 );
    else
      out = zeros( [ sImg nCoils ] );
      out( sampleMask == 1 ) = in;
    end
  end

  acr = cropData( kData, [ sACR(1) sACR(2) nCoils ] );
  if doFindGamma == true
    [w, gamma] = findW( acr, wSize );
  else
    w = findW( acr, wSize );
  end

  if numel( gamma ) == 0
    gamma = zeros( nCoils, 1 );
  elseif isscalar( gamma )
    gamma = gamma * ones( nCoils, 1 );
  end

  flipW = padData( flipDims( w, 'dims', [1 2] ), [ sImg nCoils nCoils ] );
  function out = applyW( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = sum( circConv2( flipW, in ), 3 );
      out = squeeze( out );
    else
      out = zeros( sImg(1), sImg(2), nCoils );
      for c = 1 : nCoils
        out = out + circConv2( flipW(:,:,:,c), in(:,:,c), 'transp' );
      end
    end
  end

  if lambda > 0
    sqrtLambda = sqrt( lambda );
  else
    sqrtLambda = 1;
  end

  function out = applyWmI( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = sqrtLambda * ( applyW( in ) - in );
    else
      out = sqrtLambda * ( applyW( in, 'transp' ) - in );
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

  function out = applyFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      Sin = bsxfun( @times, sMaps, in );
      out = fftshift2( fft2( ifftshift2( Sin ) ) );
    else
      FAhIn = fftshift2( fft2h( ifftshift2( in ) ) );
      out = sum( conj(sMaps) .* FAhIn, 3 );
    end
  end

  function out = applyMFSPT( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyM( applyFS( applyP( in, 'transp' ) ) );
    else
      out = applyP( applyFS( applyM( in, 'transp' ), 'transp' ) );
    end
  end

  function out = applyA( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      if numel( support ) > 0
        in = applyP( in, 'transp' );
      else
        in = reshape( in, sImg );
      end
      FSin = applyFS( in );
      MFSin = applyM( FSin );
      WmIFSin = applyWmI( FSin );
      out = [ MFSin(:); WmIFSin(:); ];
    else
      MhIn = applyM( in( 1 : nSamples ), op );
      in_bottom = reshape( in(nSamples+1:end), size( kData ) );
      WmIhin_bottom = applyWmI( in_bottom, op );
      AhIn = MhIn + WmIhin_bottom;
      out = applyFS( AhIn, 'transp' );
      if numel( support ) > 0
        out = out( support == 1 );
      else
        out = out(:);
      end
    end
  end

  if doChecks == true
    sKData = size( kData );
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

    if numel( support ) > 0
      [chkP,errChkP] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyP );
      if chkP == true
        disp( 'Check of P Adjoint passed' );
      else
        error([ 'Check of P Adjoint failed with error ', num2str(errChkP) ]);
      end
    end

    [chkFS,errChkFS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyFS );
    if chkFS == true
      disp( 'Check of FS Adjoint passed' );
    else
      error([ 'Check of FS Adjoint failed with error ', num2str(errChkFS) ]);
    end

    if numel( support ) > 0
      [chkMFSPT,errChkMFSPT] = checkAdjoint( rand( nSupport, 1 ) + 1i * rand( nSupport, 1 ), @applyMFSPT );
    else
      [chkMFSPT,errChkMFSPT] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyMFSPT );
    end
    if chkMFSPT == true
      disp( 'Check of MFSPT Adjoint passed' );
    else
      error([ 'Check of MFSPT Adjoint failed with error ', num2str(errChkMFSPT) ]);
    end

    if numel( support ) > 0
      [chkA,errChkA] = checkAdjoint( rand( nSupport, 1 ) + 1i * rand( nSupport, 1 ), @applyA );
    else
      [chkA,errChkA] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyA );
    end
    if chkA == true
      disp( 'Check of A Adjoint passed' );
    else
      error([ 'Check of A Adjoint failed with error ', num2str(errChkA) ]);
    end
  end

  if numel( support ) > 0
    img0 = zeros( nSupport, 1 );
  else
    img0 = zeros( sImg );
  end

  maxGamma = max( gamma );
  if maxGamma > 0  &&  lambda == 0
    img = mri_reconSpiritReg_minOverSphere( kData, gamma, @applyA, img0 );

  elseif maxGamma == 0  &&  lambda > 0
    img = mri_reconSpiritReg_tikhonov( kData, @applyA, img0 );

  elseif lambda == 0  &&  maxGamma == 0
    %img = mri_reconModelBased( kData, 'sMaps', sMaps, 'doCheckAdjoint', doChecks );
    b = kData( sampleMask == 1 );
    tol = 1d-6;
    nMaxIter = 100;
    [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( applyA, b, tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>

  else  % lambda > 0  &&  maxGamma > 0
    error( 'gamma and lambda cannot both be set.' );
  end

  if numel( support ) > 0
    img = applyP( img, 'transp' );
  else
    img = reshape( img, sImg );
  end
end

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

function img = mri_reconSpiritReg_minOverSphere( kData, gammas, applyA, img0, varargin )
  % minimize (1/2) || M F S x - b ||_2^2
  % subject to || ( W - I ) F S(c) x ||_2^2 / nPix <= gamma for each c
  %   where nPix is the number of pixels in the image

  p = inputParser;
  p.addParameter( 'optAlg', 'pdhgWLS', @(x) true );
  p.parse( varargin{:} );
  optAlg = p.Results.optAlg;

  f = @(in) 0;
  proxf = @(in,t) in;

  % g(in1,in2) = (1/2) || in1 - b ||_2^2 + sum_c indicator( || in2(:,:,c) ||_Fro^2 / nPix <= gammas(c) )
  % g(in1,in2) = g1( in1 ) + g2( in2 );

  b = kData( kData ~= 0 );
  nSamples = numel( b );
  sKData = size( kData );
  nPix = sKData(1) * sKData(2);
  nCoils = size( kData, 3 );
  g1 = @(in1) 0.5 * norm( in1 - b, 2 ).^2;

  function out = g2( in2 )
    in2 = reshape( in2, sKData );
    for coilIndx = 1 : nCoils
      if norm( in2(:,:,coilIndx), 'fro' )^2 / nPix > gammas(coilIndx)
        out = Inf;
        return
      end
    end
    out = 0;
  end

  function out = g( in )
    in1 = in( 1 : nSamples );
    in2 = in( nSamples + 1 : end );
    out = g1( in1 ) + g2( in2 );
  end

  function out = proxg1( in1, t )
    out = proxL2Sq( in1, t, b );
  end

  function out = proxg2( in2 )
    % indicator( || in2 ||_2^2 / nPix <= gammas(coilIndx) ) is equivalent to
    % indicator( || in2(:,:,coilIndx) ||_Fro <= sqrt( gammas( coilIndx ) * nPix ) )
    in2 = reshape( in2, sKData );
    out = zeros( sKData );
    for coilIndx = 1 : nCoils
      normIn2Coil = norm( in2(:,:,coilIndx), 'fro' );
      ballRadius = sqrt( gammas( coilIndx ) * nPix );
      if normIn2Coil <= ballRadius
        out(:,:,coilIndx) = in2(:,:,coilIndx);
      else
        out(:,:,coilIndx) = ( ballRadius / normIn2Coil ) * in2(:,:,coilIndx);
      end
    end
    out = out(:);
  end

  function out = proxg( in, t )
    in1 = in( 1 : nSamples );
    in2 = in( nSamples + 1 : end );
    out = [ proxg1( in1, t ); proxg2( in2 ); ];
  end

  proxgConj = @(in,s) proxConj( @proxg, in, s );

  sImg = size( kData, [1 2] );
  %normA = powerIteration( applyA, rand(sImg) + 1i * rand( sImg ) );
load( 'normA.mat', 'normA' );
  tau = 1 / normA;

  verbose = true;

  if strcmp( optAlg, 'pdhg' )
    [img, objValues, relDiffs] = pdhg( img0(:), proxf, proxgConj, tau, 'A', applyA, 'f', f, 'g', @g, ...
      'N', 1000, 'normA', normA, 'printEvery', 10, 'verbose', verbose );   %#ok<ASGLU>
  elseif strcmp( optAlg, 'pdhgWLS' )
    [img, objValues] = pdhgWLS( img0(:), proxf, proxgConj, 'A', applyA, 'tau', tau, ...
      'f', f, 'g', @g, 'verbose', verbose );   %#ok<ASGLU>
  else
    error( 'Unrecognized optimization algorithm' );
  end
end

function img = mri_reconSpiritReg_tikhonov( kData, applyA, img0 )

  b = [ kData( kData ~= 0 ); zeros( numel( kData ), 1 ); ];

  tol = 1d-6;
  nMaxIter = 1000;
  [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( applyA, b, tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>
end
