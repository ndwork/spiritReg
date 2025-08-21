
function run_reconSpiritRegCS
  close all; rng(1);

  addpath( './dworkLib' );

  verbose = true;
  printEvery = 20;
  wSize = 5;
  minSACR = 20;
  sampleFractions = fliplr( [ 0.12 0.15 0.2 0.25 0.3 0.35 ] );
  outDir = './out';
  diffScale = 8;

  gammas = [ 0.1 0.5 1 2 4 5 10 ];
  lambdas = 10 .^ ( -2 : 1 );

  if ~exist( outDir, 'dir' ), mkdir( outDir ); end
  logFile = [ outDir, filesep(), 'log.csv' ];
  if ~exist( logFile, 'file' )
    logID = fopen( logFile, 'a' );
    fprintf( logID, 'datacase, sampleFraction, sampleBurden, type, ssim, psnr, nIterations, lambda, gamma \n' );
    fclose( logID );
  end

  nLambdas = numel( lambdas );
  nGammas = numel( gammas );

  load( 'datacase9', 'kData', 'noiseCoords' );
  kData = squeeze( kData ) / max( abs( kData(:) ) );
  sImg = size( kData, [ 1 2 ] );
  [ ~, sACR ] = makeWavSplit( sImg );
  sACR = max( sACR, minSACR );
  nCoils = size( kData, 3 );

  for sampleFractionIndx = 1 : numel( sampleFractions )
    sampleFraction = sampleFractions( sampleFractionIndx );
    disp([ 'Working on sample fraction ', num2str( sampleFraction) ]);

    datacaseOutDir = [ outDir, filesep, 'ankle' ];
    thisOutDir = [ datacaseOutDir, filesep, 'sf_', num2str(sampleFraction) ];
    if ~exist( thisOutDir, 'dir' )
      mkdir( thisOutDir );
      mkdir( [ thisOutDir, filesep, 'pics' ]);
      mkdir( [ thisOutDir, filesep, 'pics_wSupport' ]);
      mkdir( [ thisOutDir, filesep, 'spiritRegCS' ]);
      mkdir( [ thisOutDir, filesep, 'spiritRegCSwSupport' ]);
    end

    sampleMaskFile = [ thisOutDir, filesep, 'sampleMask.mat' ];
    if exist( sampleMaskFile, 'file' )
      load( sampleMaskFile, 'sampleMaskCS' );
    else
      nSamples = round( sampleFraction * size( kData, 1 ) * size( kData, 2 ) );
      acrMask = padData( ones( sACR ), sImg );
      sampleMaskCS = mri_makeSampleMask( sImg, nSamples, 'maskType', 'VDPD', 'startMask', acrMask, 'Delta', 0.5 );
      save( sampleMaskFile, 'sampleMaskCS' );
    end

    kSamplesCS = bsxfun( @times, kData, sampleMaskCS );

    sampleBurden = sum(sampleMaskCS(:)) / numel( sampleMaskCS(:) );
    disp([ 'Sample burden with CS: ', num2str( sampleBurden ) ]);

    acr = cropData( kSamplesCS, [ sACR nCoils ] );
    [ sMaps, support ] = callPISCO( acr, sImg, 'tau', 3 );

    imgTrue = mri_reconModelBased( kData, sMaps );
    maxImgTrue = max( abs( imgTrue(:) ) );
    scaledImgTrue = imgTrue / maxImgTrue;

    trueImgOut = [ datacaseOutDir, filesep, 'trueImg.png' ];
    if ~exist( trueImgOut, 'file' )
      imwrite( abs(scaledImgTrue), trueImgOut );
      trueMatOut = [ datacaseOutDir, filesep, 'trueImg.mat' ];
      save( trueMatOut, 'imgTrue' );
    end

    noiseValues = imgTrue( noiseCoords(2):noiseCoords(4), noiseCoords(1):noiseCoords(3) );
    epsSupport = norm( noiseValues(:), 2 )^2 / numel( noiseValues );

    %parfor paramIndx = 1 : nLambdas * nGammas * 2
    for paramIndx = 1 : nLambdas * nGammas * 2
      [ lambdaIndx, gammaIndx ] = ind2sub( [ nLambdas nGammas*2 ], paramIndx );
      lambda = lambdas( lambdaIndx );   %#ok<PFBNS>

      if gammaIndx > nGammas
        gamma = gammas( gammaIndx - nGammas );   %#ok<PFBNS>
        type = 'spiritRegCSwSupport';
      else
        gamma = gammas( gammaIndx );
        type = 'spiritRegCS';
      end

      lamOutDir = [ thisOutDir, filesep, type, filesep, 'lam_', num2str(lambda) ];
      if ~exist( lamOutDir, 'dir' ), mkdir( lamOutDir ); end
      imgOut = [ lamOutDir, filesep, 'gam_', num2str( gamma ), '.png' ];
      matOut = [ lamOutDir, filesep, 'gam_', num2str( gamma ), '.mat' ];
      if exist( imgOut, 'file' )
        disp([ 'Already complete: ', imgOut ]);
        loaded = load( matOut );
        imgRecon = loaded.imgRecon;
        objValues = loaded.objValues;

      else
        disp([ 'Working on lambdaIndx / gammaIndx: ', num2str( lambdaIndx ), ' / ', num2str( gammaIndx ) ]);

        if gammaIndx <= nGammas
          [ imgRecon, objValues ] = mri_reconPICS( kSamplesCS, sMaps, 'wSize', wSize, 'sACR', sACR, ...
            'lambda', lambda, 'gamma', gamma, 'printEvery', printEvery, 'verbose', verbose );
        else
          [ imgRecon, objValues ] = mri_reconPICS( kSamplesCS, sMaps, 'wSize', wSize, 'sACR', sACR, ...
            'lambda', lambda, 'gamma', gamma, 'support', support, 'epsSupport', epsSupport, ...
            'printEvery', printEvery, 'verbose', verbose );
        end
      end

      nIterations = numel( objValues );
      scaledRecon = imgRecon / maxImgTrue;
      ssimRecon = ssim( abs( scaledImgTrue ), abs( scaledRecon ) );
      psnrRecon = psnr( scaledImgTrue, scaledRecon );
      logID = fopen( logFile, 'a' );
      fprintf( logID, [ 'ankle, %.6f, %.6f,', type, ', %.6f, %.6f, %i, %.6f, %.6f \n' ], ...
        sampleFraction, sampleBurden, ssimRecon, psnrRecon, nIterations, lambda, gamma );
      fclose( logID );
      imwrite( abs( scaledRecon ), imgOut );
      parsave( matOut, imgRecon, objValues );
      diffOut = [ lamOutDir, filesep, 'diff_gam_', num2str( gamma ), '.png' ];
      imwrite( abs( scaledRecon - scaledImgTrue ) * diffScale, diffOut )
      diffOut2 = [ lamOutDir, filesep, 'diff2_gam_', num2str( gamma ), '.png' ];
      imwrite( abs( scaledRecon - scaledImgTrue ) * diffScale*2, diffOut2 )
    end

    for lambdaIndx = 1 : nLambdas * 2
      if lambdaIndx <= nLambdas
        lambda = lambdas( lambdaIndx );
        type = 'pics';
      else
        lambda = lambdas( lambdaIndx - numel( lambdas ) );
        type = 'pics_wSupport';
      end

      imgOut = [ thisOutDir, filesep, type, filesep, 'lam_', num2str(lambda), '.png' ];
      matOut = [ thisOutDir, filesep, type, filesep, 'lam_', num2str(lambda), '.mat' ];
      if exist( imgOut, 'file' )
        disp([ 'Already complete: ', imgOut ]);
        loaded = load( matOut );
        imgRecon = loaded.imgRecon;
        objValues = loaded.objValues;

      else
        disp([ 'Working on lambdaIndx ', num2str( lambdaIndx ) ]);

        if lambdaIndx <= numel( lambdas )
          [ imgRecon, objValues ] = mri_reconPICS( kSamplesCS, sMaps, 'lambda', lambda, ...
            'printEvery', printEvery, 'verbose', verbose );
        else
          [ imgRecon, objValues ] = mri_reconPICS( kSamplesCS, sMaps, 'lambda', lambda, ...
            'support', support, 'epsSupport', epsSupport, 'printEvery', printEvery, 'verbose', verbose );
        end
      end

      nIterations = numel( objValues );
      scaledRecon = imgRecon / maxImgTrue;
      ssimRecon = ssim( abs( scaledImgTrue ), abs( scaledRecon ) );
      psnrRecon = psnr( scaledImgTrue, scaledRecon );
      logID = fopen( logFile, 'a' );
      fprintf( logID, [ '%i, %.6f, %.6f, ', type, ', %.6f, %.6f, %i, %.6f \n' ], datacase, sampleFraction, sampleBurden, ...
        ssimRecon, psnrRecon, nIterations, lambda );
      fclose( logID );
      imwrite( abs(scaledRecon), imgOut );
      parsave( matOut, imgRecon, objValues );
      diffOut = [ thisOutDir, filesep, type, filesep, 'diff_lam_', num2str(lambda), '.png' ];
      imwrite( abs( scaledRecon - scaledImgTrue ) * diffScale, diffOut )
      diffOut2 = [ thisOutDir, filesep, type, filesep, 'diff2_lam_', num2str(lambda), '.png' ];
      imwrite( abs( scaledRecon - scaledImgTrue ) * diffScale * 2, diffOut2 )
    end

  end
end
