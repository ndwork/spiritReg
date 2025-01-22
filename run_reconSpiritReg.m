function run_reconSpiritReg( datacases )
  close all; rng(1);

  showScale = 5;
  verbose = true;
  wSize = 5;
  sACR = 19;
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
    load( 'stuff.mat', 'kData', 'sampleMask', 'sMaps', 'support' );
    kData = squeeze( kData ) / max( abs( kData(:) ) );  % assumes one slice.
    nCoils = size( kData, 3 );

    nSamples = round( sampleFraction * size( kData, 1 ) * size( kData, 2 ) );
    sImg = size( kData, [ 1 2 ] );
    acrMask = padData( ones( sACR, sACR ), sImg );
    %sampleMask = mri_makeSampleMask( sImg, nSamples, 'maskType', 'VDPD', 'startMask', acrMask );
    sampleMask = ones( size( kData, [1 2] ) );
    sampleMask(1:2:end,:) = 0;
    sampleMask(:,1:2:end) = 0;
    sampleMask = sampleMask | acrMask;
    kSamples = bsxfun( @times, kData, sampleMask );

    %acr = cropData( kData, [ sACR sACR nCoils ] );
    %[ sMaps, support ] = callPISCO( acr, sImg );

    imgTrue = mri_reconRoemer( mri_reconIFFT( kData, 'multiSlice', true ), 'sMaps', sMaps );
    figure;  imshowscale( abs( imgTrue ), showScale );  titlenice( 'Truth' );

    %imgMBR = mri_reconModelBased( kSamples, 'sMaps', sMaps );
    %figure;  imshowscale( abs( imgMBR ), showScale );  titlenice( 'MBR' );

    %imgSpirit = mri_reconSpirit( kSamples, sACR, wSize );
    %figure;  imshowscale( abs( imgSpirit ), showScale );  titlenice( 'Spirit' )

    %imgSpiritRegLam = mri_reconSpiritReg( kSamples, sACR, wSize, sMaps, 'lambda', 0 );
    %figure;  imshowscale( abs( imgSpiritRegLam ), showScale );  titlenice( 'SpiritRegLam' )

    %imgSpiritRegGam = mri_reconSpiritReg( kSamples, sACR, wSize, sMaps, 'gamma', 1d-3 / numel( kSamples ) );
    imgSpiritRegGam = mri_reconSpiritReg( kSamples, sACR, wSize, sMaps, 'support', support );
    figure;  imshowscale( abs( imgSpiritRegGam ), showScale );  titlenice( 'SpiritRegGam' );

    disp( 'I got here' );
  end
end
