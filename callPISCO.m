
function [ senseMaps, eig_mask ] = callPISCO( kCal, dim_sens, varargin )
  
  p = inputParser;
  p.addParameter( 'M', [], @ispositive );
  p.addParameter( 'PowerIteration_G_nullspace_vectors', 0 );
  p.addParameter( 'threshold_mask', [], @isnonnegative );
  p.addParameter( 'verbose', false );
  p.parse( varargin{:} );
  M = p.Results.M;
  PowerIteration_G_nullspace_vectors = p.Results.PowerIteration_G_nullspace_vectors;
  threshold_mask = p.Results.threshold_mask;
  verbose = p.Results.verbose;

  tau = 3; % Kernel radius

  %threshold = 0.08; % Theshold for C-matrix singular values
  threshold = []; % Theshold for C-matrix singular values
  
  if numel( M ) == 0
    M = 20; % Number of iteration for Power Iteration
  end
  
  PowerIteration_flag_convergence = []; % If equal to 1 a convergence error 
  %                                      is displayed for Power Iteration if 
  %                                      the method has not converged for 
  %                                      some voxels after the iterations 
  %                                      indicated by the user. In this example
  %                                      it corresponds to an empty array which 
  %                                      indicates that the default value is 
  %                                      being used, which is equal to 1.
  
  PowerIteration_flag_auto = 1; % If equal to 1 Power Iteration is run until
  %                               convergence in case the number of
  %                               iterations indicated by the user is too
  %                               small. 
  %                               In this example this variable corresponds 
  %                               to an empty array which indicates that the
  %                               default value is being used, which 
  %                               is equal to 0.
  
  interp_zp = []; % Amount of zero-padding to create the low-resolution grid 
  %                 if FFT-interpolation is used. In this example it
  %                 corresponds to an empty array which indicates that the
  %                 default value is being used.
  
  gauss_win_param = []; %     Parameter needed for the Gaussian apodizing 
  %                           window used to generate the low-resolution 
  %                           image in the FFT-based interpolation approach.
  %                           This corresponds to the reciprocal value of
  %                           the standard deviation of the Gaussian window. 
  %                           In this example it corresponds to an empty 
  %                           array which indicates that the default value is 
  %                           being used.

  kernel_shape = 1; % An ellipsoidal shape is adopted for the calculation of 
  %                   kernels (instead of a rectangular shape)
  
  FFT_nullspace_C_calculation = 1; % FFT-based calculation of nullspace 
  %                                  vectors of C by calculating C'*C directly
  %                                  (instead of calculating C first)

  
  if numel( PowerIteration_G_nullspace_vectors ) == 0
    PowerIteration_G_nullspace_vectors = 0; % If 1, a PowerIteration approach is used
    %                                         to find nullspace vectors of the 
    %                                         G matrices (instead of using SVD)
  end
  
  
  FFT_interpolation = 1; % Sensitivity maps are calculated on a small spatial
  %                        grid and then interpolated to a grid with nominal 
  %                        dimensions using an FFT-approach

  if numel( threshold_mask ) == 0
    threshold_mask = 0.1;
  end

  if numel( verbose ) == 0
    verbose = 0; % If equal to 1 then PISCO information is displayed
  end


  [senseMaps, eigenValues] = PISCO_senseMaps_estimation(kCal,dim_sens,...
      tau, ...
      threshold, ...
      kernel_shape, ...
      FFT_nullspace_C_calculation, ...
      PowerIteration_G_nullspace_vectors, M, PowerIteration_flag_convergence, PowerIteration_flag_auto, ...
      FFT_interpolation, interp_zp, gauss_win_param, ...
      verbose);

  eig_mask = eigenValues(:,:,end) < threshold_mask;
end