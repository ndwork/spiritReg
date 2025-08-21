
function [img, objValues] = mri_reconPICS( kData, sMaps, varargin )
  % [img, objValues, mValues] = mri_reconSpiritRegCS( kData, varargin )
  % 
  % Reconstructs the image by solving the following optimization problem:
  % minimize (0.5) || M F S x - b ||_2^2 + lambda || Psi x ||_1 
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

  p = inputParser;
  p.addParameter( 'doChecks', false );
  p.addParameter( 'epsSupport', [] );
  p.addParameter( 'gamma', [] );
  p.addParameter( 'lambda', 1, @isnonnegative );
  p.addParameter( 'N', 1000, @(x) ispositive(x) || numel(x) == 0 );
  p.addParameter( 'printEvery', 20, @ispositive );
  p.addParameter( 'sACR', [] );
  p.addParameter( 'support', [] );
  p.addParameter( 'verbose', true );
  p.addParameter( 'wSize', [] );
  p.parse( varargin{:} );
  doChecks = p.Results.doChecks;
  epsSupport = p.Results.epsSupport;
  gamma = p.Results.gamma;
  lambda = p.Results.lambda;
  N = p.Results.N;
  printEvery = p.Results.printEvery;
  sACR = p.Results.sACR;
  support = p.Results.support;
  verbose = p.Results.verbose;
  wSize = p.Results.wSize;

  if isscalar( sACR ), sACR = [ sACR sACR ]; end
  if isscalar( wSize ), wSize = [ wSize wSize ]; end

  sKData = size( kData );
  nKData = numel( kData );
  nCoils = sKData( 3 );
  sImg = sKData(1:2);
  sampleMask  =  kData ~= 0;
  if numel( sACR ) == 0
    [ wavSplit, sACR ] = makeWavSplit( sImg );
  else
    wavSplit = makeWavSplit( sImg );
  end


  %-- Nested function definitions

  function out = applyM( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( sampleMask == 1 );
    else
      out = zeros( sKData );
      out( sampleMask == 1 ) = in;
    end
  end

  function out = applyMFS( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = applyM( applyF( applyS( in ) ) );
    else
      out = applyS( applyF( applyM( in, op ), op ), op );
    end
  end

  function out = applyPc( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = in( support == 0 );
    else
      out = zeros( sImg );
      out( support == 0 ) = in;
    end
  end

  function out = Psi( in, op )
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

  function out = applyS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = bsxfun( @times, sMaps, in );
    else
      out = sum( conj(sMaps) .* in, 3 );
    end
  end

  function out = applySw( in, op )   %#ok<INUSD>
    % Apply the spirit norm weights
    % Note that since the weights are real, Sw is self adjoint
    out = spiritNormWeights .* in;
  end

  function out = applySwWmIFS( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applySw( applyWmI( applyF( applyS( in ) ) ) );
    else
      SwHIn = applySw( in, 'transp' );
      out = applyS( applyF( applyWmI( SwHIn, 'transp' ), 'transp' ), 'transp' );
    end
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
    % apply W minus I
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyW( in ) - in;
    else
      out = applyW( in, 'transp' ) - in;
    end
  end

  function out = concat_MFS_I( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      out = [ reshape( applyMFS(in), [], 1 );  in(:) ];
    else
      out = applyMFS( in(1:nb), op ) + reshape( in(nb+1:end), sImg );
    end
  end

  function out = concat_MFS_SwWmIFS( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      MFSin = applyMFS( in );
      SwWmIFSin = gamma^2 * spiritScaling * applySwWmIFS( in );
      out = [ MFSin(:); SwWmIFSin(:); ];
    else
      in1 = in( 1 : nb );
      SHFHMTin1 = applyMFS( in1, 'transp' );
      in2 = reshape( in( nb+1 : end ), sKData );
      FHSHSwWHmIin2 = gamma^2 * spiritScaling * applySwWmIFS( in2, 'transp' );
      out = SHFHMTin1 + FHSHSwWHmIin2;
    end
  end

  function out = concat_MFS_SwWmIFS_I( in, op )
    if nargin < 2  ||  strcmp( op, 'notransp' )
      MFSin = applyMFS( in );
      SwWmIFSin = gamma^2 * spiritScaling * applySwWmIFS( in );
      out = [ MFSin(:); SwWmIFSin(:); in(:); ];
    else
      in1 = in( 1 : nb );
      SHFHMTin1 = applyMFS( in1, 'transp' );
      in2 = reshape( in( nb+1 : nb+nKData ), sKData );
      FHSHSwWHmIin2 = gamma^2 * spiritScaling * applySwWmIFS( in2, 'transp' );
      in3 = in( nb+nKData+1 : end );
      out = SHFHMTin1 + FHSHSwWHmIin2 + reshape( in3, sImg );
    end
  end

  if numel( support ) > 0
    nOutsideSupport = sum( reshape( support == 0, [], 1 ) );
    outsideSupportBallRadius = sqrt( epsSupport * nOutsideSupport );
  end
  function out = indicatorOutsideSupport( in )
    out = indicatorFunction( norm( applyPc( in ) ), [ 0, outsideSupportBallRadius] );
  end

  function out = metricSupport( in )
    out = max( norm( applyPc( in ) ) - outsideSupportBallRadius, 0 );
  end

  function out = projOutsideSupportOntoBall( in )
    out = in;
    out( support == 0 ) = projectOntoBall( applyPc( in ), outsideSupportBallRadius );
  end


  %-- Checks

  img0 = mri_reconRoemer( mri_reconIFFT( kData ) );
  b = applyM( kData );
  nb = numel( b );

  if numel( wSize ) > 0
    acr = cropData( kData, [ sACR(1) sACR(2) nCoils ] );
    spiritNormWeights = findSpiritNormWeights( kData );
    spiritNormWeightsACR = cropData( spiritNormWeights, [ sACR(1) sACR(2) nCoils ] );
    w = findW( acr, wSize, spiritNormWeightsACR );
    flipW = padData( flipDims( w, 'dims', [1 2] ), [ sImg nCoils nCoils ] );

    normMFS = powerIteration( @applyMFS, rand( sImg ) );
    normSwWmIFS = powerIteration( @applySwWmIFS, rand( sImg ) );
    spiritScaling = normMFS / normSwWmIFS;
  end

  if doChecks == true
    [chkF,errChkF] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyF );
    if chkF == true
      disp( 'Check of F Adjoint passed' );
    else
      error([ 'Check of F Adjoint failed with error ', num2str(errChkF) ]);
    end

    [chkM,errChkM] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @applyM );
    if chkM == true
      disp( 'Check of M Adjoint passed' );
    else
      error([ 'Check of M Adjoint failed with error ', num2str(errChkM) ]);
    end

    [chkS,errChkS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyS );
    if chkS == true
      disp( 'Check of S Adjoint passed' );
    else
      error([ 'Check of S Adjoint failed with error ', num2str(errChkS) ]);
    end
    
    [chkMFS,errChkMFS] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyMFS );
    if chkMFS == true
      disp( 'Check of MFS Adjoint passed' );
    else
      error([ 'Check of MFS Adjoint failed with error ', num2str(errChkMFS) ]);
    end

    [ chk_MFS_I, errChk_MFS_I ] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @concat_MFS_I );
    if chk_MFS_I == true
      disp( 'Check of concat_MFS_I adjoint passed' );
    else
      error([ 'Check of concat_MFS_I adjoint failed with error ', num2str(errChk_MFS_I) ]);
    end

    [ chk_MFS_SwWmIFS, errChk_MFS_SwWmIFS ] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @concat_MFS_SwWmIFS );
    if chk_MFS_SwWmIFS == true
      disp( 'Check of concat_MFS_SwWmIFS adjoint passed' );
    else
      error([ 'Check of concat_MFS_SwWmIFS adjoint failed with error ', num2str(errChk_MFS_SwWmIFS) ]);
    end

    [ chk_MFS_SwWmIFS_I, errChk_MFS_SwWmIFS_I ] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @concat_MFS_SwWmIFS_I );
    if chk_MFS_SwWmIFS_I == true
      disp( 'Check of concat_MFS_SwWmIFS_I adjoint passed' );
    else
      error([ 'Check of concat_MFS_SwWmIFS_I adjoint failed with error ', num2str(errChk_MFS_SwWmIFS_I) ]);
    end
  end


  %-- Reconstruct the image

  
  if wSize > 0

    bw0 = [ b; zeros( nKData, 1 ) ];
    ATb = concat_MFS_SwWmIFS( bw0, 'transp' );

    if numel( support ) > 0

      % minimize (0.5) || M F S x - b ||_2^2 + gamma/2 || Ss Sw (W - I) F S x ||_2^2 + lambda || Psi x ||_1 
      % subject to || Pc x ||_2^2 / nOutsideSupport <= epsSupport;

      f = @(in) lambda * norm( reshape( Psi( in ), [], 1 ), 1 );
      proxf = @(in,t) proxCompositionAffine( @proxL1Complex, in, @Psi, 0, 1, t * lambda );
  
      applyA = @concat_MFS_SwWmIFS_I;
  
      g1 = @(in) 0.5 * norm( in - bw0, 'fro' )^2;
      proxg1 = @(in,t) proxL2Sq( in, t, bw0 );
  
      g2 = @indicatorOutsideSupport;
      proxg2 = @(in,t) projOutsideSupportOntoBall( in );
  
      g = @(in) g1( in(1:nb+nKData) ) + g2( in(nb+nKData+1:end) );
      proxg = @(in,t) [ proxg1( in(1:nb+nKData), t );  proxg2( in(nb+nKData+1:end), t ); ];
      proxgConj = @(in,s) proxConj( proxg, in, s );
  
      PsiImg0 = Psi( img0 );
      tau0 = mean( abs( PsiImg0(:) ) ) / lambda;
  
      metric1 = @(in) g1( concat_MFS_SwWmIFS( in ) );
      metric2 = @(in) f( in );
      metric3 = @metricSupport;
      metrics = { metric1, metric2, metric3 };
      metricNames = { 'dc', 'sparsity', 'supportVio' };
  
      if nargout > 1
        [ img, objValues ] = pdhgWLS( img0, proxf, proxgConj, 'A', applyA, 'f', f, 'g', g, 'tau', tau0, 'N', N, ...
          'metrics', metrics, 'metricNames', metricNames, 'printEvery', printEvery, 'verbose', verbose );
      else
        img = pdhgWLS( img0, proxf, proxgConj, 'A', applyA, 'tau', tau0, 'N', N, ...
          'metrics', metrics, 'metricNames', metricNames, 'printEvery', printEvery, 'verbose', verbose );
      end

    else

      % minimize (0.5) || M F S x - b ||_2^2 + gamma/2 || Ss Sw (W - I) x ||_2^2 + lambda || Psi x ||_1 
      % minimize (0.5) || A x - (b,0) ||_2^2 + lambda || Psi x ||_1

      g = @(in) 0.5 * norm( concat_MFS_SwWmIFS( in ) - bw0 )^2;
      gGrad = @(in) concat_MFS_SwWmIFS( concat_MFS_SwWmIFS( in ), 'transp' ) - ATb;

      h = @(in) lambda * norm( reshape( Psi(in), [], 1 ), 1 );      
      proxth = @(in,t) proxCompositionAffine( @proxL1Complex, in, @Psi, 0, 1, t * lambda );
    
      PsiImg0 = Psi( img0 );
      t0 = mean( abs( PsiImg0(:) ) ) / lambda;
    
      if nargout > 1
        [img,objValues] = fista_wLS( img0, g, gGrad, proxth, 'h', h, 'N', N, 't0', t0, 'tol', 0, ...
          'printEvery', printEvery, 'verbose', verbose );
      else
        img = fista_wLS( img0, g, gGrad, proxth, N', N, 't0', t0, 'tol', 0, ...
          'printEvery', printEvery, 'verbose', verbose );
      end


    end

  else

    if numel( support ) > 0
      % minimize (0.5) || M F S x - b ||_2^2 + lambda || Psi x ||_1
      % subject to || Pc x ||_2^2 / nOutsideSupport <= epsSupport;
  
      f = @(in) lambda * norm( reshape( Psi( in ), [], 1 ), 1 );
      proxf = @(in,t) proxCompositionAffine( @proxL1Complex, in, @Psi, 0, 1, t * lambda );
  
      applyA = @concat_MFS_I;
  
      g1 = @(in) 0.5 * norm( in - b, 'fro' )^2;
      proxg1 = @(in,t) proxL2Sq( in, t, b );
  
      g2 = @indicatorOutsideSupport;
      proxg2 = @(in,t) projOutsideSupportOntoBall( in );
  
      g = @(in) g1( in(1:nb) ) + g2( in(nb+1:end) );
      proxg = @(in,t) [ proxg1( in(1:nb), t );  proxg2( in(nb+1:end), t ); ];
      proxgConj = @(in,s) proxConj( proxg, in, s );
  
      PsiImg0 = Psi( img0 );
      tau0 = mean( abs( PsiImg0(:) ) ) / lambda;
  
      metric1 = @(in) g1( applyMFS( in ) );
      metric2 = @(in) f( in );
      metric3 = @metricSupport;
      metrics = { metric1, metric2, metric3 };
      metricNames = { 'dc', 'sparsity', 'supportVio' };
  
      if nargout > 1
        [ img, objValues ] = pdhgWLS( img0, proxf, proxgConj, 'A', applyA, 'f', f, 'g', g, 'tau', tau0, 'N', N, ...
          'metrics', metrics, 'metricNames', metricNames, 'printEvery', printEvery, 'verbose', verbose );
      else
        img = pdhgWLS( img0, proxf, proxgConj, 'A', applyA, 'tau', tau0, 'N', N, ...
          'metrics', metrics, 'metricNames', metricNames, 'printEvery', printEvery, 'verbose', verbose );
      end
  
    else
  
      % minimize (0.5) || M F S x - b ||_2^2 + lambda || Psi x ||_1
    
      g = @(in) 0.5 * norm( applyMFS( in ) - b )^2;
      h = @(in) lambda * norm( reshape( Psi(in), [], 1 ), 1 );
    
      SHFHMTb = applyMFS( b, 'transp' );
      gGrad = @(in) applyMFS( applyMFS( in ), 'transp' ) - SHFHMTb;
      proxth = @(in,t) proxCompositionAffine( @proxL1Complex, in, @Psi, 0, 1, t * lambda );
    
      PsiImg0 = Psi( img0 );
      t0 = mean( abs( PsiImg0(:) ) ) / lambda;
    
      if nargout > 1
        [img,objValues] = fista_wLS( img0, g, gGrad, proxth, 'h', h, 'N', N, 't0', t0, 'tol', 0, ...
          'printEvery', printEvery, 'verbose', verbose );
      else
        img = fista_wLS( img0, g, gGrad, proxth, N', N, 't0', t0, 'tol', 0, ...
          'printEvery', printEvery, 'verbose', verbose );
      end
  
    end

  end

end



%% SUPPORT FUNCTIONS

function out = applyF( in, op )
  if nargin < 2 || strcmp( op, 'notransp' )
    out = fftshift2( fft2( ifftshift2( in ) ) );
  elseif strcmp( op, 'inv' )
    out = fftshift2( ifft2( ifftshift2( in ) ) );
  elseif strcmp( op, 'invTransp' )
    out = fftshift2( ifft2h( ifftshift2( in ) ) );
  elseif strcmp( op, 'transp' )
    out = fftshift2( fft2h( ifftshift2( in ) ) );
  else
    error( 'unrecognized operation' );
  end
end


function spiritNormWeights = findSpiritNormWeights( kData )

  sampleMask = kData(:,:,1) ~= 0;
  nCoils = size( kData, 3 );

  fftCoords = size2fftCoordinates( size( kData, [1 2] ) );
  ky = fftCoords{1};
  kx = fftCoords{2};
  [ kxs, kys ] = meshgrid( kx, ky );
  kDists = sqrt( kxs.*kxs + kys.*kys );
  sampleMask( kDists == 0 ) = 0;
  kDistsSamples = kDists( sampleMask == 1 );
  [ kDistsSamples, sortedIndxs ] = sort( kDistsSamples );

  spiritNormWeights = cell( 1, 1, nCoils );
  kDistThresh = 0.075;
  parfor coilIndx = 1 : nCoils
    kDataC = kData(:,:,coilIndx);
    F = kDataC( kData(:,:,coilIndx) ~= 0 );
    F = F( sortedIndxs );

    simpleNormWeights = false;
    if simpleNormWeights == true

      coeffs = fitPowerLaw( kDistsSamples, abs( F ), 'ub', [Inf 0], 'linear', false );   %#ok<PFBNS>
      m = coeffs(1);
      p = coeffs(2);
      normWeights = ( kDists.^(-p) ) / m;

    else

      freqBreak = 0.1;

      % Find the parameters that fit the power law for low frequencies    
      coeffs = fitPowerLaw( kDistsSamples( kDistsSamples <= freqBreak ), ...
                            abs( F( kDistsSamples <= freqBreak ) ), 'ub', [Inf 0], 'linear', false );   %#ok<PFBNS>
      mLow = coeffs(1);
      pLow = coeffs(2);
      %fitLow = mLow * kDistsSamples.^pLow;
      normWeightsLow = ( kDists.^(-pLow) ) / mLow;

      % Find the parameters tha tfit the power law for high frequencies
      coeffs = fitPowerLaw( kDistsSamples( kDistsSamples >= freqBreak ), ...
                            abs( F( kDistsSamples >= freqBreak ) ), 'ub', [Inf 0], 'linear', false );   %#ok<PFBNS>
      mHigh = coeffs(1);
      pHigh = coeffs(2);
      %fitHigh = mHigh * kDistsSamples.^pHigh;
      normWeightsHigh = ( kDists.^(-pHigh) ) / mHigh;

      %figure;  plotnice( kDistsSamples, log( abs( F ) ) );
      %hold on;  plotnice( kDistsSamples, log( fitLow ) );
      %hold on;  plotnice( kDistsSamples, log( fitHigh ) );

      % Blend the parameters together

      %fitBlend = max( fitLow, fitHigh );
      %hold on;  plotnice( kDistsSamples, log( fitBlend ) );
      %figure;  plotnice( kDists(:), normWeights(:) );
      normWeights = min( normWeightsLow, normWeightsHigh );

    end

  
    % fit a line to very small kDists and find the y intercept in order to set the weights when kDists == 0
    smallKDists = kDists( kDists < kDistThresh  &  kDists ~= 0 );
    smallKNormWeights = normWeights( kDists < kDistThresh  &  kDists ~= 0 );
    polyCoeffs = fitPolyToData( 1, smallKDists, smallKNormWeights );
    normWeights( kDists == 0 ) = polyCoeffs(1);

    spiritNormWeights{ coilIndx } = normWeights / sum( normWeights(:) );
  end
  spiritNormWeights = cell2mat( spiritNormWeights );
end


function [w, epsSpiritReg] = findW( acr, wSize, spiritNormWeights )
  %-- Find the interpolation coefficients w

  sACR = size( acr );
  nCoils = sACR( 3 );

  w = cell( 1, 1, 1, nCoils );
  epsSpiritReg = zeros( nCoils, 1 );
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
        if numel( spiritNormWeights ) > 0
          b( aIndx ) = spiritNormWeights(j,i) * b( aIndx );
          A( aIndx, : ) = spiritNormWeights(j,i) * A( aIndx, : );
        end
        aIndx = aIndx + 1;
      end
    end

    wCoil = A \ b(:);
    epsSpiritReg( coilIndx ) = norm( A * wCoil - b )^2 / numel( b );
    wCoil = [ wCoil( 1 : pt2RemoveIndx - 1 ); 0; wCoil( pt2RemoveIndx : end ); ];
    w{1,1,1,coilIndx} = reshape( wCoil, [ wSize nCoils ] );
  end
  w = cell2mat( w );
end