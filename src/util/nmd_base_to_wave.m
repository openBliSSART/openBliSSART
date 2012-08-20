function [] = nmd_base_to_wave(matrix_file, wave_file_prefix, windowsize, overlap, Fs)

if (nargin < 3)
    windowsize = 1024;
end
if (nargin < 4)
    overlap = 0.75;
end
if (nargin < 5)
    Fs = 16000;
end
    
shift = (1.0 - overlap) * windowsize;
data = readblissart(matrix_file);
window = sqrt(hanning(windowsize));

for ci = 1:size(data, 2)
    % Change the order of dimensions to create a spectrogram X of the
    % W(p), p = 1 ... P.
    Xi = reshape(data(:,ci,:), size(data, 1), size(data, 3));
    N = size(Xi, 2);
    % number of samples
    n = (N-1) * shift + windowsize;
    xi = zeros(n, 1);
    for j = 1:N
        spec = Xi(:,j);
        % Force symmetric DFT (assuming conjugate complex data of size
        % 'windowsize')
        frame = ifft(spec, windowsize, 'symmetric');
        % Window reconstructed frames to reduce discontinuities at frame
        % borders.
        frame = frame .* window;
        start = (j-1) * shift + 1;
        frameend = start + windowsize - 1;
        xi(start:frameend) = xi(start:frameend) + frame;
    end
    wavwrite(xi, Fs, strcat(wave_file_prefix, '_', int2str(ci), '.wav'));
end



end