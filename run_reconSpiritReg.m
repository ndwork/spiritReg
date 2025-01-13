function run_reconSpiritReg( datacases )
  close all; rng(1);

  showScale = 3;
  verbose = true;
  wSize = 5;
  sACR = 31;
  sampleFraction = 0.3;

  if nargin < 1
    %datacases = [ 7 11 1 0 2:6 8:10 ];
    datacases = 9;
  end

  [~, nCores] = getNumCores();
  nWorkers = nCores - 4;
  %if ~parpoolExists() && nWorkers > 1, parpool( nWorkers ); end

  nDatacases = numel( datacases );
  for datacaseIndx = 1 : nDatacases
    close all;
    datacase = datacases( datacaseIndx );
    disp([ 'Working on datacase ', indx2str( datacase, max( datacases ) ) ]);

    %[ kData, noiseCoords, ~, trueRecon ] = loadDatacase( datacase );
    load( 'stuff.mat', 'kData', 'sampleMask', 'sMaps' );
    kData = squeeze( kData ) / max( abs( kData(:) ) );  % assumes one slice.

    nSamples = round( sampleFraction * size( kData, 1 ) * size( kData, 2 ) );
    sImg = size( kData, [ 1 2 ] );
    acrMask = padData( ones( sACR, sACR ), sImg );
    %sampleMask = mri_makeSampleMask( sImg, nSamples, 'maskType', 'VDPD', 'startMask', acrMask );
    sampleMask = ones( size( kData, [1 2] ) );
    sampleMask(:,1:3:end) = 0;
    sampleMask(:,2:3:end) = 0;
    sampleMask = sampleMask | acrMask;
    kSamples = bsxfun( @times, kData, sampleMask );

    %sMaps = mri_makeSensitivityMaps( kSamples );

    img = mri_reconSpiritReg( kSamples, sACR, wSize, 'lambda', 1d-3, 'sMaps', sMaps );
    %img = mri_reconSpirit( kSamples, sACR, wSize, 'epsilon', 1 );

    figure;  imshowscale( abs( img ) , 3 );
    disp( 'I got here' );
  end
end
