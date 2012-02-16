function data = readblissart(filename, precision)

fp = fopen(filename, 'r');
if (nargin < 2)
    precision = 'double';
end
type = fread(fp, 1, 'uint32');
if (type ~= 2)
    error('not a blissart matrix file')
end
rows = fread(fp, 1, 'uint32');
cols = fread(fp, 1, 'uint32');
data = fread(fp, [rows, cols], precision);
fclose(fp);

end