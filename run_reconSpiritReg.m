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
load( 'stuff.mat', 'kData', 'sampleMask', 'noiseCoords' ); 
    kData = squeeze( kData ) / max( abs( kData(:) ) );  % assumes one slice.
    nCoils = size( kData, 3 );

    nSamples = round( sampleFraction * size( kData, 1 ) * size( kData, 2 ) );
    sImg = size( kData, [ 1 2 ] );
    acrMask = padData( ones( sACR, sACR ), sImg );
    %sampleMask = mri_makeSampleMask( sImg, nSamples, 'maskType', 'VDPD', 'startMask', acrMask );
    sampleMask = ones( size( kData, [1 2] ) );
    sampleMask(1:3:end,:) = 0;
    sampleMask(2:3:end,:) = 0;
    sampleMask(:,1:2:end) = 0;
    sampleMask = sampleMask | acrMask;
    kSamples = bsxfun( @times, kData, sampleMask );

    acr = cropData( kSamples, [ sACR sACR nCoils ] );
    [ sMaps, support ] = callPISCO( acr, sImg );
    figure;  imshowscale( support, showScale );

    imgTrue = mri_reconRoemer( mri_reconIFFT( kData, 'multiSlice', true ), 'sMaps', sMaps );
    figure;  imshowscale( abs( imgTrue ), showScale );  titlenice( 'Truth' );

    noiseValues = imgTrue( noiseCoords(2):noiseCoords(4), noiseCoords(1):noiseCoords(3) );
    noiseBound = norm( noiseValues(:), 2 )^2 / numel( noiseValues );

    %imgMBR = mri_reconModelBased( kSamples, 'sMaps', sMaps );
    %figure;  imshowscale( abs( imgMBR ), showScale );  titlenice( 'MBR' );

    %imgSpirit = mri_reconSpirit( kSamples, sACR, wSize );
    %figure;  imshowscale( abs( imgSpirit ), showScale );  titlenice( 'Spirit' )

    % imgSpiritRegSptNoise = mri_reconSpiritReg( kSamples, sMaps, 'support', support, ...
    %   'noiseBound', noiseBound );
    % figure;  imshowscale( abs( imgSpiritRegSptNoise ), showScale );  titlenice( 'SpiritRegSptNoise' )

    imgSpiritRegMFSNbGam = mri_reconSpiritReg( kSamples, sMaps, 'support', support, ...
      'sACR', sACR, 'wSize', wSize, 'noiseBound', noiseBound );
    figure;  imshowscale( abs( imgSpiritRegMFSNbGam ), showScale );  titlenice( 'SpiritRegMFSNbGam' )

    imgSpiritRegMFSNb = mri_reconSpiritReg( kSamples, sMaps, 'support', support, ...
      'noiseBound', noiseBound );
    figure;  imshowscale( abs( imgSpiritRegMFSNb ), showScale );  titlenice( 'SpiritRegMFSNb' )

    imgSpiritRegMFSPTGam = mri_reconSpiritReg( kSamples, sMaps, 'support', support, ...
      'sACR', sACR, 'wSize', wSize );
    figure;  imshowscale( abs( imgSpiritRegMFSPTGam ), showScale );  titlenice( 'SpiritRegMFSPTGam' )

    imgSpiritRegGam = mri_reconSpiritReg( kSamples, sMaps, 'wSize', wSize, 'sACR', sACR );
    figure;  imshowscale( abs( imgSpiritRegGam ), showScale );  titlenice( 'SpiritRegGam' )

    imgSpiritRegMFSPT = mri_reconSpiritReg( kSamples, sMaps, 'support', support );
    figure;  imshowscale( abs( imgSpiritRegMFSPT ), showScale );  titlenice( 'SpiritRegMFSPT' )

    imgSpiritRegMFS = mri_reconSpiritReg( kSamples, sMaps );
    figure;  imshowscale( abs( imgSpiritRegMFS ), showScale );  titlenice( 'SpiritRegMFS' )

    %imgSpiritRegGam = mri_reconSpiritReg( kSamples, sACR, wSize, sMaps, 'gamma', 1d-3 / numel( kSamples ) );
    imgSpiritRegGam = mri_reconSpiritReg( kSamples, sMaps, ...
      'sACR', sACR, 'support', support, 'wSize', wSize, 'noiseBound', noiseBound );
    figure;  imshowscale( abs( imgSpiritRegGam ), showScale );  titlenice( 'SpiritRegGam' );

    kSamplesAug = fftshift2( fft2( ifftshift2( bsxfun( @times, sMaps, imgSpiritRegGam ) ) ) );
    [ ~, support2 ] = callPISCO( kSamplesAug, sImg );
    figure;  imshowscale( support2, showScale );
    imgSpiritRegGam2 = mri_reconSpiritReg( kSamples, sMaps, 'sACR', sACR, 'support', support2, ...
      'wSize', wSize, 'noiseBound', noiseBound );
    figure;  imshowscale( abs( imgSpiritRegGam2 ), showScale );  titlenice( 'SpiritRegGam2' );

    disp( 'I got here' );
  end
end
