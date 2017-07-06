function [y, v] = vl_nnhebbian(x, v, varargin)

opts.lambda = 0.1 ;
opts.eta = 0.0005 ;
opts.mode = 'train' ;
opts.pass = 'forward' ;
opts.dzdx = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if strcmp(opts.pass, 'backward')
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
    if isempty(dzdy)
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
    else
        x = sparse(double(x)) ;
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
        x = single(full(x)) ;
        
        old_v = full(v) ;
        v = old_v + opts.eta * mean((reshape(x,[1, size(x, 1), size(x, 2)]) ...
          .* reshape(x, [size(x, 1), 1, size(x, 2)])) .* double(old_v ~= 0), 3) ;
        v(v > 1) = 1 ; v(v < -1) = -1 ;
        v = sparse(double(v)) ;
        
%         x = single(full(x)) ;
%         y = single(full(y)) ;
        
%         figure(2) ;
%         clf ;
%         histogram(v(v ~= 0), linspace(-1, 1, 150), 'FaceColor', 'm') ;
%         xlim([-1, 1]) ;
%         ylim([0, 5000]) ;
%         
%         % xlim([weight_counter - 15, weight_counter]) ;
%         % ylim([-1, 1]) ;
%         % plot(ones(numel(v), 1) * weight_counter, v(:), '.b', 'MarkerSize', 1) ;
%         
%         pause(0.01) ;
%         hold on ;
    end
else
    if nargin <= 1 || isempty(dzdy)
        x = sparse(double(x)) ;
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
        x = single(full(x)) ;
        y = single(full(y)) ;
    else
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
    end
end

x = reshape(x, orig_size) ;
y = reshape(y, orig_size) ;