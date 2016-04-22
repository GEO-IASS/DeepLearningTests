classdef ptp
    methods(Static)
        %format of winners: {architectures{labels{plotdata(ddbn),plotdata(mlp)}}}
        %on the first level winners contains cells for each architecure
        %the architecture cell contains cells for each label
        %the label contains a cell of the format
        %{plotdata(ddbn),plotdata(mlp)} where the ddbn and mlp have comparable results
        function winners = run(base)
            architecture_folders = ptp.getFolders(base);
            winners = {};
            for architecture_folder = architecture_folders
                winners{end + 1} = ptp.run_architecture(base,architecture_folder);
            end
        end
        
        %architectures is a cell array with arrays of the hidden layers to filter
        %labels is an array of labels to filter
        %winners is of the format returned from ptp.run
        function filtered_winners = filter_winners(hidden_layers,labels,winners)
            filtered_winners = {};
            for i = 1:size(winners,2)
               hidden_layer = winners{i}{1}{1}.hidden_layers;
               if isempty(find(cellfun(@(c) isequal(c,hidden_layer),hidden_layers)))
                   filtered_labels = {};
                   for j = 1:size(winners{i},2)
                       if not(isequal(winners{i}{j},struct))
                           label = winners{i}{j}{1}.label;
                           if isempty(find(arrayfun(@(c) isequal(c,label),labels)))
                               filtered_labels{end + 1} = winners{i}{j};
                           end
                       end
                   end
                   filtered_winners{end + 1} = filtered_labels;
               end
            end
        end
        
        function createBarMultiplot(base,winners)
            fig = figure('visible','off');
            x = {};
            y = [];
            for i = 1:size(winners,2)
                architecture = winners{i};
                %x{end + 1} = mat2str(architecture{1}{1});
                label = architecture{i};
                yRow = zeros(1,size(architecture,2));
                for j = 1:size(architecture,2)    
                    if iscell(label)
                        ddbn = label{1};
                        mlp = label{2};
                        
                    end
                end
            end
        end
        %Prints the given plotData in one plot with a row for each
        %architecture and a column for each label.
        %The format of winners is the same as the return value from ptp.run
        function createMultiplot(base,winners,transpose)
            fig = figure('Visible','off');
            if transpose
                plot_rows = size(winners{1},2);
                plot_columns = size(winners,2);
                get_subplot_index = @(i,j) (j-1)*plot_columns + i;
            else
                plot_rows = size(winners,2);
                plot_columns = size(winners{1},2);
                get_subplot_index = @(i,j) (i-1)*plot_columns + j;
            end
            for i = 1:size(winners,2)
                architecture = winners{i};
                for j = 1:size(architecture,2)
                    label = architecture{j};
                    if iscell(label)
                        ddbn = label{1};
                        mlp = label{2};
                        subplot(plot_rows,plot_columns,get_subplot_index(i,j));
                        plot(fliplr(mlp.epochs),mlp.val_ers,'r'...
                            ,fliplr(ddbn.epochs),ddbn.val_ers,'g'); %...
                            %,fliplr(mlp.epochs),mlp.train_ers,'r--'...
                            %,fliplr(ddbn.epochs),ddbn.train_ers,'g--'...
                        %);
                        title(sprintf('%s %g',mat2str(ddbn.hidden_layers),ddbn.label));
                        %legend('mlp validation','mlp train','ddbn validation','ddbn train');                    
                    end
                end
            end
            plot_path = sprintf('%s/plot.png',base);
            saveas(fig,plot_path,'png');
        end
                
        function winners =  run_architecture(base,architecture_folder)
            base = fullfile(base,architecture_folder.name);
            lable_folders = ptp.getFolders(base);
            winners = {};
            for label_folder = lable_folders
                winners{end + 1} = ptp.run_label(base,label_folder,architecture_folder.name);
            end
            ptp.createMultiplot(base,{winners},1);
        end
        
        function res = run_label(base,label_folder,architecture)
            base = fullfile(base,label_folder.name);
            settings_folders = ptp.getFolders(base);
            label = str2double(label_folder.name(4:end));
            for settings_folder = settings_folders
                res = ptp.run_settings(base,settings_folder,label,architecture);
            end
        end
        
        function res = run_settings(base,settings_folder,label,architecture)
            base = fullfile(base,settings_folder.name);
            mat_files = ptp.getMatFiles(base);
            [ddbns,mlps,~] = ptp.split_and_load_DDBN_MLP_OTHER(base,mat_files);
            if not(length(ddbns) == 0) && not(length(mlps) == 0)
                best_ddbn = ptp.getBestNetwork(ddbns);
                best_mlp = ptp.getBestNetwork(mlps);
                fig = ptp.plotNetworkComparison(best_ddbn,best_mlp);
                if ptp.areComparable(best_ddbn,best_mlp)
                    plot_path = ptp.getPlotPath(base,architecture,label,best_ddbn,best_mlp);
                    saveas(fig,plot_path,'png');
                    res = {plotInfo(best_ddbn,label) plotInfo(best_mlp,label)};
                else
                    warning('two incomparable networks found in %s',base);
                    res = struct;
                end
            else
                res = struct;
            end
        end
        
        function fig = plotNetworkComparison(ddbn,mlp)
            fig = figure('Visible','off');
            plot(fliplr(mlp.epochs),mlp.val_ers,'r'...
                ,fliplr(mlp.epochs),mlp.train_ers,'r--'...
                ,fliplr(ddbn.epochs),ddbn.val_ers,'g'...
                ,fliplr(ddbn.epochs),ddbn.train_ers,'g--');
            legend('mlp validation','mlp train','ddbn validation','ddbn train');
            xlabel('epochs');
            ylabel('error');
        end
        
        function res = areComparable(ddbns,mpls)
            res = 1;
        end
        
        function plot_path = getPlotPath(base,architecture,label,ddbn,mlp)
            plot_path = fullfile(base...
                ,sprintf('Rectangle %s lb=%g ddbn_e = (%g,%g) mlp_e = (%g,%g).png'...
                ,architecture,label...
                ,ddbn.nn.test_er,ddbn.nn.val_er,mlp.nn.test_er,mlp.nn.val_er));
        end
        
        function res = isDirectory(pathStruct)
            res = pathStruct.isdir && not(strcmp(pathStruct.name,'.'))...
                && not(strcmp(pathStruct.name,'..'));
        end
        
        function res = isMatFile(pathStruct)
            [~,~,ext] = fileparts(pathStruct.name);
            res = not(pathStruct.isdir) && strcmp(ext,'.mat');
        end
        
        function [ddbns,mlps,others] = split_and_load_DDBN_MLP_OTHER(base,mat_files)
            ddbns = {};
            mlps = {};
            others = {};
            for mat_file = mat_files
                mat_file_contents = load(fullfile(base,mat_file.name));
                if strncmpi(mat_file.name,'MLP',3)
                    mlps{end + 1} = mat_file_contents.nn_res;
                elseif strncmpi(mat_file.name,'DDBN',4)
                    ddbns{end + 1} = mat_file_contents.dbn_res;
                else
                    others{end + 1} = mat_file_contents;
                end
            end
            ddbns = [ddbns{:}];
            mlps = [mlps{:}];
            others = [others{:}];
        end
        
        function best_network = getBestNetwork(networks)
            best_network = networks(1);
            for network = networks
                if best_network.nn.val_er > network.nn.val_er
                    best_network = network;
                end
            end
        end
        
        function mat_files = getMatFiles(path)
            contents = dir(path)';
            mat_files = {};
            for content = contents
                if ptp.isMatFile(content)
                    mat_files{end + 1} = content;
                end
            end
            mat_files = [mat_files{:}];
        end
        
        function folders = getFolders(path)
            contents = dir(path)';
            folders = {};
            for content = contents
                if ptp.isDirectory(content)
                    folders{end + 1} = content;
                end
            end
            folders = [folders{:}];
        end
    end
end

