function [y, v] = vl_nnhebbian3(x, v, indices, varargin)

opts.lambda = 0.05 ;
opts.eta = 0.0001 ;
opts.beta = 100 * opts.eta ;
opts.alpha = 1.0 ;
opts.connectivity = '8-lattice' ;
opts.mode = 'train' ;
opts.pass = 'forward' ;
opts.update = 'oja' ;
opts.do_plot = true ;
opts.save_updates = false ;
opts.save_weights = true ;
opts.dzdx = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if strcmp(opts.pass, 'forward') && strcmp(opts.mode, 'train')
    assignin('base', 'weight_counter', evalin('base', 'weight_counter') + 1) ;
end

weight_counter = evalin('base', 'weight_counter') ;

netname = ['cifar_hebbian_' num2str(opts.lambda) '_' num2str(opts.eta) ...
                                        '_' opts.connectivity] ;

update_path = fullfile('..', 'work', 'updates', netname) ;
weight_path = fullfile('..', 'work', 'weights', netname) ;
            
if strcmp(opts.pass, 'backward')
  dzdy = opts.dzdx ;
else
  dzdy = [] ;
end

orig_size = size(x) ;

x = reshape(x, numel(x(:, :, :, 1)), numel(x(1, 1, 1, :))) ;

if ~isempty(dzdy)
    dzdy = reshape(dzdy, numel(dzdy(:, :, :, 1)), numel(dzdy(1, 1, 1, :))) ;
end

if strcmp(opts.mode, 'train')
    if strcmp(opts.pass, 'forward')
        y = x ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            y(idxs(2), :) = y(idxs(2), :) + opts.lambda * v(i) * x(idxs(1), :) ;
        end
        
        % do Hebbian weight update
        old_v = v ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
                        
            switch opts.update
                case 'standard'
                    % standard Hebbian update
                    update = opts.eta * mean(x(idxs(1, :)) .* y(idxs(2, :)), 2) ;
                case 'oja'
                    % Oja's learning rule
                    update = opts.eta * mean((x(idxs(1), :) .* y(idxs(2, :)) - (y(idxs(2, :)) ^ 2 * v(i))), 2) ;
                otherwise
                    warning('Unexpected update type.')
            end
            
            v(i) = old_v(i) + update - opts.beta ;
        end
        
        v(v > 1) = 1 ; v(v < 0) = 0 ;
        
        if opts.save_weights
            if weight_counter == 1
                if exist(weight_path, 'dir') == 7   
                    delete(fullfile(weight_path, '*'))
                    rmdir(weight_path)
                end
                
                mkdir(weight_path)
            end
            
            save(fullfile(weight_path, ['weights_' num2str(weight_counter)]), 'v') ;
        end
                
        if opts.do_plot
            subplot(5, 1, 1) ;
            cla ;
            histogram(old_v, linspace(-1, max(old_v), 50), 'FaceColor', 'm') ;
            xlim([0, 1]) ;
            ylim([0, numel(old_v(old_v ~= 0)) / 2]) ;
            title('Hebbian weights') ;
            
            weight_update = v - old_v ;

            subplot(5, 1, 2) ;
            cla ;
            histogram(weight_update(weight_update ~= 0), linspace(-1, 1, 150), 'FaceColor', 'k') ;
            xlim([-1, 1]) ;
            ylim([0, numel(v) / 2]) ;
            title('Hebbian weight update') ;
            
            subplot(5, 1, 3) ;
            cla ;
            histogram(x(x ~= 0), linspace(-100, 100, 150), 'FaceColor', 'c') ;
            ylim([0, numel(x) / 2]) ;
            title('Input activations') ;

            subplot(5, 1, 4) ;
            cla ;
            histogram(y(y ~= 0), linspace(-100, 100, 150), 'FaceColor', 'bl') ;
            ylim([0, numel(x) / 2]) ;
            title('Output activations') ;

            subplot(5, 1, 5) ;
            plot(ones(1, 1) * weight_counter, sum(y(:) - x(:)), 'or') ;
            xlim([weight_counter - 25, weight_counter]) ;
            title('Total difference in activations (output - input)') ;

            pause(0.01) ;
            hold on ;
        end
    else
        % calculate input derivatives
        y = dzdy ;
        
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            y(idxs(2), :) = y(idxs(2), :) + ...
                    (y(idxs(2), :) .* opts.lambda * v(i)) ; % * x(idxs(2), :)) ;
        end
        
        % calculate weight derivatives
        dvdz = dzdy ;
        
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            dvdz(idxs(2), :) = dvdz(idxs(2), :) + ...
                (dvdz(idxs(2), :) .* (opts.lambda * x(idxs(1), :))) ;
        end
        
        diff = dvdz - y ;
        grad_update = [] ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            grad_update(end + 1) = sum(diff(idxs(2), :)) ;
        end
        
        if opts.save_updates
            if weight_counter == 1
                if exist(update_path, 'dir') == 7   
                    delete(fullfile(update_path, '*'))
                    rmdir(update_path)
                end
                
                mkdir(update_path)
            end
            
            save(fullfile(update_path, ['gradients_' num2str(weight_counter)]), 'grad_update') ;
            save(fullfile(update_path, ['hebbian_' num2str(weight_counter)]), 'weight_update') ;
        end
    end
else
    if strcmp(opts.pass, 'forward')
        y = x ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            y(idxs(2), :) = y(idxs(2), :) + opts.lambda * ...
                          v(i) * (x(idxs(1), :) .* x(idxs(2), :)) ;
        end
    else
        y = dzdy ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            y(idxs(2), :) = y(idxs(2), :) + opts.lambda ...
                                        * v(i) * x(idxs(1), :) ;
        end
    end
end

y = reshape(y, orig_size) ;
