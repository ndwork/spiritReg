

function [ data, noiseCoords, res, trueRecon ] = loadDatacase( datacase, varargin )
  %
  % Index:
  % datacase - integer specifying datacase
  %
  % Ouptuts:
  % data - array of data in hybrid state
  % res - the intended resolution corresponding to 0.5 in k-space
  % noiseCoords - [ minX minY maxX maxY ] specifying noise region in data
  %
  % Written by Nicholas Dwork, Copyright 2019

  p = inputParser;
  p.addOptional( 'loadData', true, @islogical );
  p.addParameter( 'simCase', 2, @isnumeric );
  p.parse( varargin{:} );
  loadData = p.Results.loadData;
  simCase = p.Results.simCase;
  data = [];

  dataDir = '/Volumes/NDWORK128GB/';
  if ~exist( dataDir, 'dir' )
    dataDir = '/Volumes/DataDrive/';
  end
  if ~exist( dataDir, 'dir' )
    dataDir = '/Users/nicholasdwork/.DellEMS/Vols/disk2s1/';
  end
  if ~exist( dataDir, 'dir' )
    dataDir = '/Users/ndwork/Documents/Data/';
  end
  if ~exist( dataDir, 'dir' )
    dataDir = '/Users/nicholasdwork/Documents/Data/';
  end

  res = 1d-3;  %#ok<NASGU>

  trueRecon = [];

  switch datacase

    case 0
      % Simulated datacase
      [ data, trueRecon ] = loadSimData( simCase );

      noiseCoords = [ 1 1 25 25 ];
      res = 1.0d-3;  % meters per pixel

    case 1
      if loadData == true
        data = readOldMriDataOrgData( [ dataDir, '/mriData.Org/P14/kspace' ] );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data( :, :, 5:10:end, : );
        data = data( :, :, 4, : );
      end
      noiseCoords = [ 1 1 41 76 ];
      res = 0.5d-3;  % meters per pixel

    case 2
      if loadData == true
        data = readOldMriDataOrgData( [ dataDir, '/mriData.Org/P17/kspace' ] );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data( :, :, 5:10:end, : );
        data = data( :, :, 4, : );
      end
      noiseCoords = [ 1 1 56 72 ];
      res = 0.5d-3;  % meters per pixel

    case 3
      if loadData == true
        load( './ESPIRIT/data/brain_8ch.mat', 'DATA' );
        data = permute( DATA, [1 2 4 3] );
      end
      noiseCoords = [ 1 1 35 32 ];
      res = 1.0d-3;  % meters per pixel

    case 4
      if loadData == true
        load( './ESPIRIT/data/brain_32ch.mat', 'DATA' );
        data = permute( DATA, [1 2 4 3] );
      end
      noiseCoords = [ 1 1 35 32 ];
      res = 1.0d-3;  % meters per pixel

    case 5
      % FOV 25.6 cm, 256 x 256 (1mm in plane, 1.5mm thick slices)
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/brain3d/P73216.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,64,:);
      end
      noiseCoords = [ 1 1 50 50 ];
      res = 1.0d-3;  % meters per pixel

    case 6
      % FOV 25.6 cm, 256 x 256 (1mm in plane, 1mm thick slices)
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/brain3d/P73728.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,64,:);
      end
      noiseCoords = [ 1 1 50 50 ];
      res = 1.0d-3;  % meters per pixel

    case 7
      % FOV 20.5, 256 x 256 (0.8mm in plane, 0.8mm thick slices)
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/brain3d/P74240.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,64,:);
      end
      noiseCoords = [ 1 1 45 45 ];
      res = 0.8d-3;  % meters per pixel

    case 8
      % FOV 20 cm, 400 x 400 (0.5mm in plane, 0.5mm thick slices)
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/brain3d/P74752.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,64,:);
      end
      noiseCoords = [ 1 1 70 70 ];
      res = 0.5d-3;  % meters per pixel

    case 9
      % Ankle, 256 x 256
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/body3d/P26112.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = rot90( data(:,:,280,:), -1 );
      end
      noiseCoords = [ 10 10 80 80 ];
      res = 1.0d-3;  % meters per pixel

    case 10
      % shoulder, 
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/body3d/P23040.7' ] );   %#ok<ASGLU>
        data = squeeze( rot90( data ) );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,245,:);
        data = cropData( data, [384 320 1 16] );
      end
      noiseCoords = [ 10 10 60 60 ];
      res = 1.0d-3;  % meters per pixel

    case 11
      % Ankle, 256 x 256
      if loadData == true
        [data,header] = read_MR_rawdata( [ dataDir, '/fullySampled3D/body3d/P27648.7' ] );   %#ok<ASGLU>
        data = squeeze( data );
        %data = rot90( data, -1 );
        data = ifft( ifftshift( data, 3 ), [], 3 );
        data = data(:,:,50,:);
      end
      noiseCoords = [ 10 10 100 100 ];
      res = 1.0d-3;  % meters per pixel
      
    otherwise
      error( 'This datacase doesn''t exist' );
  end

end

