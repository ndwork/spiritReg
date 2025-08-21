
function out = indx2str( indx, N )
  % out = indx2str( indx [, N ] )
  %
  % Converts index to a string the appropriate size for N with padded zeros
  % For example, if N is 100, will convert indx=88 into '088';
  %
  % Inputs:
  % indx - the number to convert into a string
  %
  % Optional Inputs:
  % N - the maximum indx that will be converted
  %
  % Outputs:
  % out - string
  %
  % Example:
  %
  % N = 1000;
  % for i=1:N
  %   disp([ 'Working on ', indx2str(i,N), ' of ', N ]);
  %   ... do important stuff here ...
  % end
  %
  % Written by Nicholas - Copyright 2019
  %
  % This software is offered under the GNU General Public License 3.0.  It
  % is offered without any warranty expressed or implied, including the
  % implied warranties of merchantability or fitness for a particular
  % purpose.

  if nargin < 1, disp( 'Usage:  out = indx2str( indx [, N ] )' ); end

  if nargin < 2
    if mod( indx, 10 ) == 0
      nDigits = floor( log10( indx ) ) + 1;
    else
      nDigits = floor( log10( indx ) ) + 2;
    end
  else
    nDigits = floor( log10( N ) ) + 1;
  end

  formatSpec = [ '%', num2str(nDigits), '.', num2str(nDigits), 'i' ];

  out = num2str( indx, formatSpec );
end
