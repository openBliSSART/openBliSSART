function data = readblissart(filename, precision)

fp = fopen(filename, 'r');
if (nargin < 2)
    precision = 'double';
end
type = fread(fp, 1, 'uint32');
if (type == 3)
    nm = fread(fp, 1, 'uint32');
else
    if (type == 2)
        nm = 1;
    else
        error('not a blissart matrix file')
    end
end
rows = fread(fp, 1, 'uint32');
cols = fread(fp, 1, 'uint32');
if (nm > 1)
    data = zeros(rows, cols, nm);
    for mi = 1:nm
        data(:, :, mi) = fread(fp, [rows, cols], precision);
    end
else
    data = fread(fp, [rows, cols], precision);
end
fclose(fp);

end
