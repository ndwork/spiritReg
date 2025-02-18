function img = mri_reconCS( kData, varargin )
  % img = mri_reconCS( kData [, 'Psi', Psi, 'sMaps', sMaps, 'support', support, 'verbose', verbose ] )
  %
  % By default, for single coil
  %   minimizes || Psi x ||_1 subject to M F x = b
  %
  % If the support is provided, then
  %   minimizes || Psi PT x ||_1 subject to M F PT x = b
  %   where P extracts the pixels within the support into a column vector
  %
  % If epsilon is provided, then it imposes the constraint
  %   minimizes || Psi x ||_1 subject to || M F x - b ||_2^2 / nb <= epsilon
  %
  % For multi coil
  %   minimize || Psi x ||_{2,1} subject to M F x(c) = b(c)
  %   where c = 1, 2, ... C is the coil index
  %
  % If lambda is supplied then minimizes (1/2) || M F S x - b ||_2^2 + lambda || Psi x ||_1
  %
  % If the support is also supplied, then minimizes (1/2) || M F S PT x - b ||_2^2 + lambda || Psi PT x ||_1
  %
  % Inputs:
  % kData - an array of size M x N x C where C is the number of coils
  %         and uncollected data have value 0.
  %
  % Optional Inputs:
  % epsilon - represents the variance of the noise
  % Psi - a function handle to an ORTHOGONAL sparsifying transformation
  %       Must operate on either a 2D image or a 3D array (a set of 2D images)
  %
  % Written by Nicholas Dwork, Copyright 2025
  %
  % https://github.com/ndwork/dworkLib.git
  %
  % This software is offered under the GNU General Public License 3.0.  It
  % is offered without any warranty expressed or implied, including the
  % implied warranties of merchantability or fitness for a particular
  % purpose.

% TODO: add preconditioning into PDHG
% TODO: warm start PDHG

  p = inputParser;
  p.addParameter( 'doChecks', false );
  p.addParameter( 'epsilon', [], @isnonnegative );
  p.addParameter( 'lambda', [], @ispositive );
  p.addParameter( 'oPsi', true );
  p.addParameter( 'optAlg', [], @(x) true );
  p.addParameter( 'Psi', [] );
  p.addParameter( 'sMaps', [], @isnumeric );
  p.addParameter( 'support', [], @(x) isnumeric(x) || islogical(x) );
  p.addParameter( 'verbose', true );
  p.parse( varargin{:} );
  doChecks = p.Results.doChecks;
  epsilon = p.Results.epsilon;
  lambda = p.Results.lambda;
  oPsi = p.Results.oPsi;
  optAlg = p.Results.optAlg;
  Psi = p.Results.Psi;
  sMaps = p.Results.sMaps;
  support = p.Results.support;
  verbose = p.Results.verbose;

  sKData = size( kData );
  sImg = sKData(1:2);
  nCoils = size( kData, 3 );
  nImg = prod( sImg );
  sampleMask = ( kData ~= 0 );
  b = kData( sampleMask == 1 );
  nb = numel( b );
  sOut = [ sImg(:)' nCoils ];
  nUnknown = prod( sOut ) - nb;

  if numel( sMaps ) > 0  &&  nCoils == 1
    error( 'Cannot provide sMaps with single coil data' );
  end

  if numel( Psi ) == 0
    wavSplit = makeWavSplit( sImg );
    if oPsi == false
      warning( 'The default value of oPsi being overwritten because default Psi is orthogonal' );
    end
    oPsi = true;
    Psi = @defaultPsi;
  end

  if numel( support ) > 0
    nSupport = sum( support(:) );
  else
    nSupport = nImg;
  end

  if oPsi ~= true
    error( 'Cannot yet handle this problem.' );
    % Look into "A Primal–Dual Splitting Method for Convex Optimization Involving 
    % Lipschitzian, Proximable and Linear Composite Terms" by Condat or the PD3O algorithms
    % to solve this problem.
  end


  %% Create linear transformations and their adjoints

  function out = defaultPsi( in, op )
    out = zeros( size( in ) );
    if nargin < 2 || strcmp( op, 'notransp' )
      for sliceIndx = 1 : size( in, 3 )
        out(:,:,sliceIndx) = wtDaubechies2( in(:,:,sliceIndx), wavSplit );
      end
    else
      for sliceIndx = 1 : size( in, 3 )
        out(:,:,sliceIndx) = iwtDaubechies2( in(:,:,sliceIndx), wavSplit );
      end
    end
  end

  function out = vPsi( in, op )
    if numel( in ) == nImg
      in = reshape( in, sImg );
    else
      in = reshape( in, sKData );
    end
    if nargin < 2 || strcmp( op, 'notransp' )
      out = Psi( in );
    else
      out = Psi( in, op );
    end
    out = out(:);
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

  function out = applyMF( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = applyM( applyF( in ) );
    else
      out = applyF( applyM( in, op ), op );
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

  function out = applyMFS( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = applyMF( applyS( in ) );
    else
      out = applyS( applyMF( in, op ), op );
    end
  end

  function out = apply_vMFS( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      in = reshape( in, sImg );
      out = applyMF( applyS( in ) );
    else
      out = applyS( applyMF( in, op ), op );
      out = out(:);
    end
  end

  function out = PsiFinvMcT( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = Psi( applyF( applyMc( in, 'transp' ), 'inv' ) );
    else
      out = applyMc( applyF( Psi( in, 'transp' ), 'transpInv' ) );
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

  function out = applyMFSPT( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = applyMFS( applyP( in, 'transp' ) );
    else
      out = applyP( applyMFS( in, op ), 'notransp' );
    end
  end

  function out = concat_MFPT_PsiPT( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      PTin = applyP( in, 'transp' );
      out1 = applyMF( PTin );
      out2 = Psi( PTin );
      out = [ out1(:); out2(:); ];
    else
      out1 = applyMF( in(1:nb), op );
      out2 = Psi( reshape( in( nb+1 : end ), sImg ), op );
      out = applyP( out1 + out2 );
    end
  end

  function out = concat_MFS_Psi( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out1 = applyMFS( in );
      out2 = Psi( in );
      out = [ out1(:); out2(:); ];
    else
      out1 = applyMFS( in(1:nb), op );
      out2 = Psi( reshape( in( nb+1 : end ), sImg ), op );
      out = out1 + out2;
    end
  end

  function out = concat_MFSPT_PsiPT( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      PTin = applyP( in, 'transp' );
      out = concat_MFS_Psi( PTin );
      %out1 = applyMFS( PTin );
      %out2 = Psi( PTin );
      %out = [ out1(:); out2(:); ];
    else
      %out1 = applyMFS( in(1:nb), op );
      %out2 = Psi( reshape( in( nb+1 : end ), sImg ), op );
      %out = applyP( out1 + out2 );
      out = applyP( concat_MFS_Psi( in, op ) );
    end
  end

doChecks = true
  if doChecks == true

    if numel( lambda ) == 0
      [chkPsiFinvMcT,errChkPsiFinvMcT] = checkAdjoint( ...
        rand( nUnknown, 1 ) + 1i * rand( nUnknown, 1 ), @PsiFinvMcT );
      if chkPsiFinvMcT == true
        disp( 'Check of PsiFinvMcT Adjoint passed' );
      else
        error([ 'Check of PsiFinvMcT Adjoint failed with error ', num2str(errChkPsiFinvMcT) ]);
      end
    end

    [chk_MF,errChk_MF] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyMF );
    if chk_MF == true
      disp( 'Check of applyMF passed' );
    else
      error([ 'Check of applyMF Adjoint failed with error ', num2str(errChk_MF) ]);
    end

    if numel( support ) > 0
      [chkP,errChkP] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyP );
      if chkP == true
        disp( 'Check of P adjoint passed' );
      else
        error([ 'Check of P adjoint failed with error ', num2str(errChkP) ]);
      end

      if nCoils == 1
        [chk_concat_MFPT_PsiPT,errChk_concat_MFPT_PsiPT] = checkAdjoint( ...
          rand( nSupport, 1 ) + 1i * rand( nSupport, 1 ), @concat_MFPT_PsiPT );
        if chk_concat_MFPT_PsiPT == true
          disp( 'Check of chk_concat_MFPT_PsiPT adjoint passed' );
        else
          error([ 'Check of chk_concat_MFPT_PsiPT failed with error ', num2str(errChk_concat_MFPT_PsiPT) ]);
        end
      end
    end

    if numel( sMaps ) > 0
      [chk_vS,errChk_vS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @apply_vS );
      if chk_vS == true
        disp( 'Check of vS Adjoint passed' );
      else
        error([ 'Check of vS Adjoint failed with error ', num2str(errChk_vS) ]);
      end

      [chk_vMFS,errChk_vMFS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @apply_vMFS );
      if chk_vMFS == true
        disp( 'Check of vMFS adjoint passed' );
      else
        error([ 'Check of vMFS adjoint failed with error ', num2str(errChk_vMFS) ]);
      end

      [chk_concat_MFS_Psi,errChk_concat_MFS_Psi] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @concat_MFS_Psi );
      if chk_concat_MFS_Psi == true
        disp( 'Check of chk_concat_MFS_Psi adjoint passed' );
      else
        error([ 'Check of chk_concat_MFS_Psi failed with error ', num2str(errChk_concat_MFS_Psi) ]);
      end

      if numel( support ) > 0
        [chkMFSPT,errChkMFSPT] = checkAdjoint( rand( nSupport, 1 ) + 1i * rand( nSupport, 1 ), @applyMFSPT );
        if chkMFSPT == true
          disp( 'Check of MFSPT adjoint passed' );
        else
          error([ 'Check of MFSPT adjoint failed with error ', num2str(errChkMFSPT) ]);
        end

        [chk_concat_MFSPT_PsiPT,errChk_concat_MFSPT_PsiPT] = checkAdjoint( ...
          rand( nSupport, 1 ) + 1i * rand( nSupport, 1 ), @concat_MFSPT_PsiPT );
        if chk_concat_MFSPT_PsiPT == true
          disp( 'Check of chk_concat_MFSPT_PsiPT adjoint passed' );
        else
          error([ 'Check of chk_concat_MFSPT_PsiPT failed with error ', num2str(errChk_concat_MFSPT_PsiPT) ]);
        end
      end
    end

    [chk_vPsi,errChk_vPsi] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @vPsi );
    if chk_vPsi == true
      disp( 'Check of vPsi Adjoint passed' );
    else
      error([ 'Check of vPsi Adjoint failed with error ', num2str(errChk_vPsi) ]);
    end

    [chk_vPsi,errChk_vPsi] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @vPsi );
    if chk_vPsi == true
      disp( 'Check (2nd) of vPsi Adjoint passed' );
    else
      error([ 'Check (2nd) of vPsi Adjoint failed with error ', num2str(errChk_vPsi) ]);
    end

  end


  %% Perform the optimization
  f = [];  proxf = [];
  metrics = [];
  metricNames = [];
  usePDHG = false;
  useFISTA = false;

  if numel( lambda ) > 0

    if numel( support ) == 0  &&  oPsi == true
      useFISTA = true;

      if nCoils > 1
        % minimize (1/2) || M F S x - b ||_2^2 + lambda || Psi x ||_1
        applyA = @applyMFS;
        img0 = zeros( sImg );

      else
        % minimize (1/2) || M F x - b ||_2^2 + lambda || Psi x ||_1
        applyA = @applyMF;
      end

      f = @(in) 0.5 * norm( applyA(in) - b, 2 )^2;
      ATb = applyA( b, 'transp' );
      fGrad = @(in) applyA( applyA( in ), 'transp' ) - ATb;
      g = @(in) lambda * norm( vPsi( in ), 1 );
      proxg = @(in,t) proxCompositionAffine( @proxL1Complex, in, Psi, 0, 1, lambda * t );

    elseif numel( support ) == 0  &&  oPsi == false
      usePDHG = true;

      g = @(in) 0.5 * norm( in(1:nb) - b )^2 + lambda * norm( in(nb+1:end), 1 );
      proxg = @(in,t) [ proxL2Sq( in(1:nb), t, b ); proxL1Complex( in(nb+1:end), lambda * t ); ];

      if nCoils > 1
        % minimize (1/2) || M F S x - b ||_2^2 + lambda || Psi x ||_1
        applyA = @concat_MFS_Psi;
        x0 = mri_reconRoemer( applyF( kData, 'inv' ) );

      else
        error( 'Not yet implemented' );
      end

    elseif numel( support ) > 0
      usePDHG = true;

      g = @(in) 0.5 * norm( in(1:nb) - b )^2 + lambda * norm( in(nb+1:end), 1 );
      proxg = @(in,t) [ proxL2Sq( in(1:nb), t, b ); proxL1Complex( in(nb+1:end), lambda * t ); ];

      if nCoils > 1
        % minimize (1/2) || M F S PT x - b ||_2^2 + lambda || Psi PT x ||_1
        applyA = @concat_MFSPT_PsiPT;
        x0 = applyP( mri_reconRoemer( applyF( kData, 'inv' ) ) );

      else
        % minimize (1/2) || M F PT x - b ||_2^2 + lambda || Psi PT x ||_1
        applyA = @concat_MFPT_PsiPT;
        x0 = applyP( applyF( kData, 'inv' ) );
      end
    else
      error( 'Not yet implemented' );
    end
   

  else
    usePDHG = true;

    if nCoils == 1

      if numel( epsilon ) == 0  ||  epsilon == 0
        % minimize || Psi F^{-1} ( Mc^T kUnknown + kData ) ||_1  where  kData = Mc^T b
        % minimize || Psi F^{-1} Mc^T kUnknown + Psi F^{-1} kData ) ||_1
        applyA = @(in,op) PsiFinvMcT( in, op );
        bCS = Psi( applyF( kData, 'inv' ) );
        g = @(in) norm( in(:) + bCS(:), 1 );  % g(in) = || in + Psi F^{-1} kData ||_1
        proxg = @(in,t) proxL1Complex( in + bCS, t ) - bCS;
        x0 = zeros( nUnknown, 1 );

      else
        % minimize || Psi x ||_1  subject to  || M F x - b ||_2^2 / nb <= epsilon
        applyA = @applyMF;
        f = @(in) norm( vPsi( in ), 1 );
        proxf = @(in,t) proxCompositionAffine( @proxL1Complex, in, Psi, 0, 1, t );
        g = @(in) indicatorFunction( norm( in - b, 2 )^2 / nb, [0 epsilon] );
        % || M F x - b ||_2^2 / nb <= epsilon  <==>  || M F x - b ||_2 <= sqrt( epsilon * nb )
        proxg = @(in,t) projectOntoBall( in - b, sqrt( epsilon * nb ) ) + b;
        x0 = applyF( kData, 'inv' );
      end

    else

      if numel( epsilon ) == 0  ||  epsilon == 0
        % minimize || Psi x ||_{2,1} subject to M F x(c) = b(c)
        % minimize || Psi x ||_{2,1} subject to Ind( || M F x - b ||_2 )
        error( 'not yet implemented' );

      else
        % minimize || Psi x ||_{2,1} subject to || M F x - b ||_2^2 / nb < epsilon
        applyA = @applyMF;
        f = @(in) normL2L1( Psi( in ) );
        proxf = @(in,t) proxCompositionAffine( @proxL2L1, in, Psi, 0, 1, t );
        g = @(in) indicatorFunction( norm( in - b, 2 )^2 / nb, [ 0 epsilon ] );
        % || M F x - b ||_2^2 / nb <= epsilon  <==>  || M F x - b ||_2 <= sqrt( epsilon * nb )
        proxg = @(in,t) projectOntoBall( in - b, sqrt( nb * epsilon ) ) + b;
        x0 = applyF( kData, 'inv' );
      end

    end
  end

  if useFISTA == true
    ATA = @(in) applyA( applyA( in ), 'transp' );
    L = powerIteration( ATA, rand( size( img0 ) ), 'symmetric', true );
    t0 = 1 / L;
    if verbose == true
      [ img, objValues, relDiffs] = fista_wLS( img0, f, fGrad, proxg, 'h', g, 't0', t0, ...
        'verbose', verbose );   %#ok<ASGLU>
    else
      img = fista_wLS( img0, f, fGrad, proxg, 'h', g, 'verbose', verbose );
    end

  elseif usePDHG == true

    proxgConj = @(in,s) proxConj( proxg, in, s );
  
    if numel( x0 ) == 0
      if nCoils > 1
        y = mri_reconRoemer( mri_reconIFFT( kData ) );
      else
        y = applyF( kData, 'inv' );
      end
    else
      normA = powerIteration( applyA, rand(size(x0)) + 1i * rand( size(x0) ) );
      tau = 1 / normA;
    
      if numel( optAlg ) == 0, optAlg = 'pdhgWLS'; end

      N = 1000;
      if strcmp( optAlg, 'pdhg' )
        [y, objValues, relDiffs] = pdhg( x0, proxf, proxgConj, tau, 'A', applyA, 'f', f, 'g', g, ...
          'dsc', true, 'N', N, 'normA', normA, 'printEvery', 1, 'verbose', verbose );   %#ok<ASGLU>
      elseif strcmp( optAlg, 'pdhgWLS' )
        [y, objValues, mValues] = pdhgWLS( x0, proxf, proxgConj, 'A', applyA, ...
          'beta', 1, 'tau', tau, 'f', f, 'g', g, 'N', N, 'dsc', true, ...
          'printEvery', 10, 'metrics', metrics, 'metricNames', metricNames, 'verbose', verbose );   %#ok<ASGLU>
      else
        error( 'Unrecognized optimization algorithm' );
      end
    end

    if size( y ) == sImg
      img = y;
    elseif numel( y ) == nUnknown
      img = applyF( applyMc( y, 'transp' ) + kData, 'inv' );
    elseif numel( y ) == nSupport
      img = applyP( y, 'transp' );
    else
      img = y;
    end

  else
    error( 'Unknown optimization method' );
  end

  if ~ismatrix( img )
    if numel( sMaps ) > 0
      img = mri_reconRoemer( img, 'sMaps', sMaps );

      applyA = @applyMFS;
      f = @(in) norm( vPsi( in ), 1 );
      proxf = @(in,t) proxCompositionAffine( @proxL1Complex, in, Psi, 0, 1, t );
      g = @(in) indicatorFunction( norm( in - b, 2 )^2 / nb, [ 0 epsilon ] );
      proxg = @(in,t) projectOntoBall( in - b, sqrt( nb * epsilon ) ) + b;
      proxgConj = @(in,s) proxConj( proxg, in, s );

      img0 = img;
      N = 2000;

      if numel( optAlg ) == 0, optAlg = 'pdhgWLS'; end

      if strcmp( optAlg, 'pdhg' )
        [img, objValues, relDiffs] = pdhg( img0, proxf, proxgConj, tau, 'A', applyA, 'f', f, 'g', g, ...
          'N', N, 'normA', normA, 'printEvery', 10, 'verbose', verbose );   %#ok<ASGLU>

      elseif strcmp( optAlg, 'pdhgWLS' )
        metric1 = f;
        metric2 = @(in) abs( norm( applyA( in ) - b, 2 ) - sqrt( nb * epsilon ) );
        metrics = { metric1, metric2 };
        metricNames = { 'sparsity', 'violation' };
        saveFun = @(in,optIter,saveDir) imwrite( abs(imresize(reshape(in,sImg),5)) / max(abs(in(:))), ...
          [ saveDir, '/pdhgWLS_save_', indx2str(optIter,N), '.png' ] );

        [img, objValues, mValues] = pdhgWLS( img0, proxf, proxgConj, 'A', applyA, 'beta', 1, 'tau', tau, ...
          'f', f, 'g', g, 'N', N, 'dsc', true, 'metrics', metrics, 'metricNames', metricNames, ...
          'verbose', verbose, 'printEvery', 10, 'saveEvery', 50, 'saveFun', saveFun, ...
          'saveDir', 'iterSaves' );   %#ok<ASGLU>
      else
        error( 'Unrecognized optimization algorithm' );
      end

    else
      img = mri_reconRoemer( img );
    end
  end
end

