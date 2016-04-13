result_files = dir('Results/Rectangles');
for result_file = result_files
    [~,name,ext] = fileparts(result_file.name);
    if strcmp(ext,'mat')
        
        fig = plot_res(
    end
end
