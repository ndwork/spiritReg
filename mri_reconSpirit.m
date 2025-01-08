
function img = mri_reconSpirit( kData, sACR, wSize, varargin )
  % img = mri_reconSpirit( kData, sACR, wSize )
  %
  % k_collected   and   k_est   <=>   k = k_collected  U  k_est
  %
  % minimize || W k - k ||_2 over k_est
  %   where k = toMatrix( D )^T k_collected + ( toMatrix( D^C ) )^T k_est
  %
  %   Here, D is a set of sample indices and k are the sample values that were collected.
  %   D^C is the complement of D
  %   That is, k = kData( D^C ).
  %
  % Equivalently, we want to minimize || ( W - I ) k ||_2 over k_est.
  % Equivalently, minimize || ( W - I ) ( toMatrix( D^C ) )^T k_est + ( W - I ) toMatrix( D )^T k_collected ||_2.
  %
  % Inputs:
  % kData - a complex matrix of size M x N x C, where C is the number of coils
  %   Uncollected data will have a value of 0.
  % sACR - a scalar or a 2 element array that specifies the size of the autocalibration region.
  %  If it's a scalar, then the size is assumed to be [ sACR sACR ].
  % wSize - either a scalar or a two element array of odd vlaues that specifies the size of the interpolation 
  %   kernel.  If wSize is a scalar, then the kernel is assumed to have size [ wSize wSize ].
  %
  % Outputs
  % img - a 2D complex array of size M x N that is the output image.

  p = inputParser;
  p.addParameter( 'doChecks', false );
  p.parse( varargin{:} );
  doChecks = p.Results.doChecks;

  if min( mod( wSize, 2 ) ) < 1, error( 'wSize elements must be odd' ); end
  if isscalar( sACR ), sACR = [ sACR sACR ]; end
  if isscalar( wSize ), wSize = [ wSize wSize ]; end

  [ M, N, nCoils ] = size( kData );
  w = zeros( wSize(1), wSize(2), nCoils, nCoils );   % Interpolation coefficients

  function out = applyW( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = zeros( M, N, nCoils );
      for c = 1 : nCoils
        tmp = circConv2( flipDims( w(:,:,:,c), 'dims', [1 2] ), in );
        out(:,:,c) = sum( tmp, 3 );
      end
    else
      out = zeros( M, N, nCoils );
      for c = 1 : nCoils
        tmp = repmat( in(:,:,c), [ 1 1 nCoils ] );
        out = out + circConv2( flipDims( w(:,:,:,c), 'dims', [1 2] ), tmp, 'transp' );
      end
    end
  end

  sampleMask = abs( kData ) > 0;
  nSamples = sum( sampleMask(:) );
  nEst = numel( sampleMask ) - nSamples;

  % b = ( W - I ) toMatrix( D )^T k_collected
  b = applyW( kData ) - kData;  b = b(:);

  % ( W - I ) ( toMatrix( D^C ) )^T k_est
  function out = applyA( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      k_est = zeros( M, N, nCoils );
      k_est( not( sampleMask ) ) = in;
      out = applyW( k_est ) - k_est;
    else
      in = reshape( in, [ M N nCoils] );
      tmp = applyW( in, 'transp' ) - in;
      out = tmp( not( sampleMask ) );
    end
    out = out(:);
  end

  if doChecks == true
    [chkW,errChkW] = checkAdjoint( rand( M, N, nCoils ) + 1i * rand( M, N, nCoils ), @applyW );
    if chkW == true
      disp( 'Check of W Adjoint passed' );
    else
      error([ 'Check of W Adjoint failed with error ', num2str(errChkW) ]);
    end

    k0 = rand( nEst, 1 ) + 1i * rand( nEst, 1 );
    [chkA,errChkA] = checkAdjoint( k0, @applyA );
    if chkA == true
      disp( 'Check of A Adjoint passed' );
    else
      error([ 'Check of A Adjoint failed with error ', num2str(errChkA) ]);
    end
  end

  %-- Find the interpolation coefficients w
yHats = zeros( (  sACR(2) - wSize(2) + 1 ) * ( sACR(1) - wSize(1) + 1 ), nCoils );
  acr = cropData( kData, [ sACR(1) sACR(2) nCoils ] );
  for coilIndx = 1 : nCoils
    A = zeros( (  sACR(2) - wSize(2) + 1 ) * ( sACR(1) - wSize(1) + 1 ), wSize(1) * wSize(2) * nCoils - 1 );
    if size( A, 1 ) < size( A, 2 ), error( 'The size of the ACR is too small for this size kernel' ); end
    y = zeros( size(A,1), 1 );
    pt2RemoveIndx = ceil( wSize(1)/2 ) + floor( wSize(2)/2 ) * wSize(1) + ( coilIndx - 1 ) * wSize(2) * wSize(1);
    aIndx = 1;
    for i = ceil( wSize(2)/2 ) : sACR(2) - floor( wSize(2)/2 )
      for j = ceil( wSize(1)/2 ) : sACR(1) - floor( wSize(1)/2 )
        subACR = acr( j - floor( wSize(2)/2 ) : j + floor( wSize(2)/2 ), ...
                               i - floor( wSize(2)/2 ) : i + floor( wSize(2)/2 ), : );
        subACR = subACR(:);
        y( aIndx ) = subACR( pt2RemoveIndx );
        subACR = [ subACR( 1 : pt2RemoveIndx-1 ); subACR( pt2RemoveIndx + 1 : end ); ];
        A( aIndx, : ) = transpose( subACR );
        aIndx = aIndx + 1;
      end
    end
    wCoil = A \ y(:);
yHat = A * wCoil;
yHats(:,coilIndx) = yHat;
yRelErr = norm( yHat(:) - y(:) ) / norm( y(:) );
disp([ 'Relative error for coil ', num2str(coilIndx) ', is: ', num2str( yRelErr ) ]);
    wCoil = [ wCoil( 1 : pt2RemoveIndx - 1 ); 0; wCoil( pt2RemoveIndx : end ); ];
    w( :, :, :, coilIndx ) = reshape( wCoil, [ wSize nCoils ] );
  end

yApplyW = applyW( kData );
yApplyW = cropData( yApplyW, [ sACR(2) - wSize(2) + 1, sACR(1) - wSize(1) + 1, nCoils ] );
yApplyW = yApplyW(:);
yApplyWRelErr = norm( yApplyW(:) - yHats(:) ) / norm( yHats(:) );
disp([ 'Relative Error for yApplyW is: ', num2str( yApplyWRelErr ) ]);


  %-- Use the interpolation coefficients to estimate the missing data

  k0 = zeros( nEst / nCoils, nCoils );
  for coilIndx = 1 : nCoils
    dataIndxs = find( sampleMask(:,:,1) ~= 0 );
    [ yDataIndxs, xDataIndxs ] = ind2sub( [ M N ], dataIndxs );
    F = scatteredInterpolant( xDataIndxs, yDataIndxs, kData( dataIndxs ), 'nearest' );
    estIndxs = find( sampleMask(:,:,1) == 0 );
    [ yEstIndxs, xEstIndxs ] = ind2sub( [ M N ], estIndxs );
    k0( :, coilIndx ) = F( xEstIndxs, yEstIndxs );
  end

  %k0 = zeros( nEst, 1 );
  tol = 1d-6;
  nMaxIter = 1000;
  [ kStar, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( @applyA, b, tol, nMaxIter, [], [], k0(:) );   %#ok<ASGLU>

  kOut = kData;
  kOut( not( sampleMask ) ) = kStar;
  img = mri_reconRoemer( mri_reconIFFT( kOut, 'dims', [ 1 2 ] ) );
end

