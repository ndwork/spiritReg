function run_reconSpiritReg( datacases )
  %close all; 
  rng(1);

  showScale = 5;
  verbose = true;
  wSize = 5;
  sACR = 19;
  sampleFraction = 0.1;

  if nargin < 1
    %datacases = [ 7 11 1 0 2:6 8:10 ];
    datacases = 9;
%datacases = 6;
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
load( 'kData.mat', 'kData', 'sampleMaskCS', 'noiseCoords' );
    kData = squeeze( kData ) / max( abs( kData(:) ) );  % assumes one slice.
    nCoils = size( kData, 3 );

    nSamples = round( sampleFraction * size( kData, 1 ) * size( kData, 2 ) );
    sImg = size( kData, [ 1 2 ] );
    nImg = prod( sImg );
    acrMask = padData( ones( sACR, sACR ), sImg );

    sampleMask = ones( size( kData, [1 2] ) );
    sampleMask(1:3:end,:) = 0;
    sampleMask(2:3:end,:) = 0;
    sampleMask(:,1:2:end) = 0;
    sampleMask = sampleMask | acrMask;
    kSamples = bsxfun( @times, kData, sampleMask );

    if ~exist( 'sampleMaskCS', 'var' )
      sampleMaskCS = mri_makeSampleMask( sImg, nSamples, 'maskType', 'VDPD', 'startMask', acrMask );
    end
    kSamplesCS = bsxfun( @times, kData, sampleMaskCS );

    disp([ 'Sample burden without CS: ', num2str( sum(sampleMask(:)) / numel( sampleMask(:) ) ) ]);
    disp([ 'Sample burden with CS: ', num2str( sum(sampleMaskCS(:)) / numel( sampleMaskCS(:) ) ) ]);

    acr = cropData( kSamples, [ sACR sACR nCoils ] );
    [ sMaps, support ] = callPISCO( acr, sImg, 'tau', 3 );
    %figure;  imshowscale( support, showScale );

    coilRecons = mri_reconIFFT( kData, 'multiSlice', true );
    noiseValues = coilRecons( noiseCoords(2):noiseCoords(4), noiseCoords(1):noiseCoords(3), : );
    noiseVar = norm( noiseValues(:), 2 )^2 / numel( noiseValues );
    noiseVars = zeros( nCoils, 1 );
    for c = 1 : nCoils
      noiseVars(c) = norm( noiseValues(:,:,c), 'fro' )^2 / numel( noiseValues(:,:,c) );
    end

    epsData = nImg * noiseVars;
    img = mriRecon( kSamplesCS, 'sMaps', sMaps, 'epsData', epsData, 'wSize', 5, 'sACR', sACR, ...
      'support', support, 'epsSupport', noiseVar );
    figure;  imshowscale( abs( img ), showScale );
    titlenice( 'spiritReg + CS w/epsData + support w/epsSupport' );

    img = mriRecon( kSamplesCS, 'sMaps', sMaps, 'epsData', noiseVar, 'wSize', 5, 'sACR', sACR );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'spiritReg + CS w/epsData' );

    img = mriRecon( kSamplesCS, 'sMaps', sMaps, 'wSize', 5, 'sACR', sACR );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'CS w/Lambda' );

    img = mriRecon( kSamplesCS, 'sMaps', sMaps, 'epsData', noiseVar );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'CS w/epsData' );

    img = mriRecon( kSamplesCS, 'sMaps', sMaps, 'lambda', 0.1 );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'CS w/Lambda' );

    img = mriRecon( kSamples, 'sMaps', sMaps );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'LSQR recon w/sMaps' );

    img = mriRecon( kData );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'Fully sampled recon' );

    img = mriRecon( kSamplesCS(:,:,1), 'support', support );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'support' );

    img = mriRecon( kSamplesCS(:,:,1) );
    figure;  imshowscale( abs( img ), showScale );  titlenice( 'support' );

  end
end
