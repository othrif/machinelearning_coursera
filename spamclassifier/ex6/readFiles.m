function file_names = readFiles(foldername)
%READFILES reads a folder and returns its entire list of contents, given the filename is of the form '*.*'
%

DirList     = dir(fullfile(foldername, '*.*'));
listOfFiles = {DirList.name};
n = 10 %length(listOfFiles);
file_names = cell(n, 1);
for j=1:n
  file_names{j} = fullfile(foldername, listOfFiles{j});
end
