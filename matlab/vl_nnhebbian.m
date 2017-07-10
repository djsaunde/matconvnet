function [y, v] = vl_nnhebbian(x, v, varargin)

opts.lambda = 0.1 ;
opts.eta = 0.0005 ;
opts.mode = 'train' ;
opts.pass = 'forward' ;
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
        % x = gather(sparse(double(x))) ;
        v = gpuArray(full(v)) ;
        
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
        % x = single(full(x)) ;
        % y = single(full(y)) ;
        
        % x = gpuArray(x) ;
        y = gpuArray(y) ;
        
        v = gather(sparse(v)) ;
                
        subplot(5, 1, 3) ;
        cla ;
        histogram(x(x ~= 0), linspace(-10, 10, 150), 'FaceColor', 'c') ;
        ylim([0, 2500]) ;
        title('Input activations') ;
        
        subplot(5, 1, 4) ;
        cla ;
        histogram(y(y ~= 0), linspace(-10, 10, 150), 'FaceColor', 'bl') ;
        ylim([0, 2500]) ;
        title('Output activations') ;
        
        subplot(5, 1, 5) ;
        plot(ones(1, 1) * weight_counter, sum(y(:) - x(:)), 'or') ;
        xlim([weight_counter - 25, weight_counter]) ;
        title('Total difference in activations (output - input)') ;
        
        pause(0.01) ;
        hold on ;
    else
        x = gather(sparse(double(x))) ;
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
        x = single(full(x)) ;
        
        old_v = full(v) ;
        weight_update = opts.eta * mean((reshape(x,[size(x, 1), 1, size(x, 2)]) ...
          .* reshape(x, [1, size(x, 1), size(x, 2)])) .* double(old_v ~= 0), 3) ;
        v = old_v + weight_update ;
        v(v > 1) = 1 ; v(v < -1) = -1 ;
        v = sparse(double(v)) ;
        
        x = gpuArray(x) ;
        y = gpuArray(y) ;
        
        subplot(5, 1, 1) ;
        cla ;
        histogram(v(v ~= 0), linspace(-1, 1, 150), 'FaceColor', 'm') ;
        xlim([-1, 1]) ;
        ylim([0, 50]) ;
        title('Hebbian weights') ;
        
        subplot(5, 1, 2) ;
        cla ;
        histogram(weight_update(weight_update ~= 0), linspace(-1, 1, 150), 'FaceColor', 'k') ;
        xlim([-1, 1]) ;
        ylim([0, 50]) ;
        title('Hebbian weight update') ;
        
        pause(0.01) ;
        hold on ;
    end
else
    if strcmp(opts.pass, 'forward')
        x = gather(sparse(double(x))) ;
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
        x = gpuArray(single(full(x))) ;
        y = gpuArray(single(full(y))) ;
    else
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
    end
end

x = reshape(x, orig_size) ;
y = reshape(y, orig_size) ;