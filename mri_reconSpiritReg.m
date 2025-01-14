
function img = mri_reconSpiritReg( kData, sACR, wSize, varargin )
  % img = mri_reconSpiritReg( kData, sACR, wSize [, 'lambda', lambda, 'sigma', sigma, 'sMaps', sMaps ] )
  %
  % minimize || M F S x - b ||_2^2 + lambda || ( W - I ) F S x ||_2^2
  % ... equivalently ...
  % minimize ||  [             M            ] F S x - [ b ] ||
  %          ||  [ sqrt( lambda ) ( W - I ) ]         [ 0 ] ||_2
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
  %
  % Written by Nicholas Dwork, Copyright 2025
  %
  % https://github.com/ndwork/dworkLib.git
  %
  % This software is offered under the GNU General Public License 3.0.  It
  % is offered without any warranty expressed or implied, including the
  % implied warranties of merchantability or fitness for a particular
  % purpose.

  %TODO: Convert spirit regularization to a spirit bound

  if nargin < 3
    disp([ 'Usage: img = mri_reconSpiritReg( kData, sACR, wSize [, ''lambda'', lambda,', ...
           ' ''sigma'', sigma, ''sMaps'', sMaps ] )' ]);
    if nargout > 0, img = []; end
    return
  end

  if min( mod( wSize, 2 ) ) < 1, error( 'wSize elements must be odd' ); end
  if isscalar( sACR ), sACR = [ sACR sACR ]; end
  if isscalar( wSize ), wSize = [ wSize wSize ]; end

  p = inputParser;
  p.addParameter( 'doChecks', false );
  p.addParameter( 'lambda', 0, @isnonnegative );
  p.addParameter( 'sMaps', [], @isnumeric );
  p.parse( varargin{:} );
  doChecks = p.Results.doChecks;
  lambda = p.Results.lambda;
  sMaps = p.Results.sMaps;

  sImg = size( kData, [ 1 2 ] );
  nCoils = size( kData, 3 );
  sampleMask = kData ~= 0;
  nSamples = nnz( sampleMask(:) );  %nnz = number of nonzero elements

  acr = cropData( kData, [ sACR(1) sACR(2) nCoils ] );
  w = findW( acr, wSize );

  %-- Use the interpolation coefficients to estimate the missing data
  function out = applyW( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = zeros( [ sImg nCoils ] );
      for c = 1 : nCoils
        tmp = circConv2( flipDims( w(:,:,:,c), 'dims', [1 2] ), in );
        out(:,:,c) = sum( tmp, 3 );
      end
    else
      out = zeros( [ sImg nCoils ] );
      for c = 1 : nCoils
        tmp = repmat( in(:,:,c), [ 1 1 nCoils ] );
        out = out + circConv2( flipDims( w(:,:,:,c), 'dims', [1 2] ), tmp, 'transp' );
      end
    end
  end

  sqrtLambda = sqrt( lambda );
  function out = A_bottom( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = sqrtLambda * ( applyW( in ) - in );
    else
      out = sqrtLambda * ( applyW( in, 'transp' ) - in );
    end
  end

  function out = A_top( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = in( sampleMask == 1 );
    else
      out = zeros( size( kData ) );
      out( sampleMask == 1 ) = in;
    end
  end

  if numel( sMaps ) == 0, sMaps = mri_makeSensitivityMaps( kData ); end

  function out = applyA( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      in = reshape( in, sImg );
      fVals = fftshift2( fft2( ifftshift2( bsxfun( @times, sMaps, in ) ) ) );
      out_top = A_top( fVals );
      out_bottom = A_bottom( fVals );
      out = [ out_top(:); out_bottom(:); ];
    else
      A_top_hIn = A_top( in( 1 : nSamples ), op );
      in_bottom = reshape( in(nSamples+1:end), size( kData ) );
      A_bottom_hIn = A_bottom( in_bottom, op );
      AhIn = A_top_hIn + A_bottom_hIn;
      FAhIn = fftshift2( fft2h( ifftshift2( AhIn ) ) );
      out = sum( bsxfun( @times, conj(sMaps), FAhIn ), 3 );
      out = out(:);
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

    [chkA_top,errChkA_top] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @A_top );
    if chkA_top == true
      disp( 'Check of A_top Adjoint passed' );
    else
      error([ 'Check of A_top Adjoint failed with error ', num2str(errChkA_top) ]);
    end

    [chkA_bottom,errChkA_bottom] = checkAdjoint( rand( sKData ) + 1i * rand( sKData ), @A_bottom );
    if chkA_bottom == true
      disp( 'Check of A_bottom Adjoint passed' );
    else
      error([ 'Check of A_bottom Adjoint failed with error ', num2str(errChkA_bottom) ]);
    end

    [chkA,errChkA] = checkAdjoint( rand( sImg ) + 1i * rand( sImg ), @applyA );
    if chkA == true
      disp( 'Check of A Adjoint passed' );
    else
      error([ 'Check of A Adjoint failed with error ', num2str(errChkA) ]);
    end
  end

  b = [ kData( sampleMask == 1 ); zeros( numel( kData ), 1 ); ];

  img0 = zeros( sImg );
  tol = 1d-6;
  nMaxIter = 1000;
  [ img, lsqrFlag, lsqrRelRes, lsqrIter, lsqrResVec ] = lsqr( @applyA, b, tol, nMaxIter, [], [], img0(:) );   %#ok<ASGLU>

  img = reshape( img, sImg );
end


function w = findW( acr, wSize )
  %-- Find the interpolation coefficients w
  sACR = size( acr );
  nCoils= sACR( 3 );

  w = cell( 1, 1, 1, nCoils );
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
    wCoil = [ wCoil( 1 : pt2RemoveIndx - 1 ); 0; wCoil( pt2RemoveIndx : end ); ];
    w{1,1,1,coilIndx} = reshape( wCoil, [ wSize nCoils ] );
  end
  w = cell2mat( w );
end
