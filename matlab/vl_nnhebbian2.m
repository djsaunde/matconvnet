function [y, v] = vl_nnhebbian2(x, v, indices, varargin)

opts.lambda = 0.1 ;
opts.eta = 0.0005 ;
opts.mode = 'train' ;
opts.pass = 'forward' ;
opts.do_plot = false ;
opts.dzdx = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if strcmp(opts.pass, 'forward')
    assignin('base', 'weight_counter', evalin('base', 'weight_counter') + 1) ;
    weight_counter = evalin('base', 'weight_counter') ;
end

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
        
        if opts.do_plot
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
        % calculate derivatives
        y = dzdy ;
        
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            y(idxs(2), :) = y(idxs(2), :) + y(idxs(2), :) .* ...
                                    (opts.lambda * v(i) * x(idxs(2), :)) ;
        end
        
        % do Hebbian weight update
        old_v = v ;
        for i = 1:length(indices)
            idxs = indices(:, i) ;
            update = opts.eta * mean(x(idxs(1), :) .* x(idxs(2), :), 2) ;
            v(i) = old_v(i) + update ;
        end
                
        % truncate values in range [-1, 1]
        v(v > 1) = 1 ; v(v < -1) = -1 ;
            
        if opts.do_plot
            subplot(5, 1, 1) ;
            cla ;
            histogram(v, linspace(-1, 1, 150), 'FaceColor', 'm') ;
            xlim([-1, 1]) ;
            ylim([0, numel(v) / 2]) ;
            title('Hebbian weights') ;
            
            weight_update = v - old_v ;

            subplot(5, 1, 2) ;
            cla ;
            histogram(weight_update(weight_update ~= 0), linspace(-1, 1, 150), 'FaceColor', 'k') ;
            xlim([-1, 1]) ;
            ylim([0, numel(v) / 2]) ;
            title('Hebbian weight update') ;

            pause(0.01) ;
            hold on ;
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

x = reshape(x, orig_size) ;
y = reshape(y, orig_size) ;