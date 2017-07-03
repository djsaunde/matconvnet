function [y, v] = vl_nnhebbian(x, v, varargin)

if ~isempty(varargin) && ~ischar(varargin{1})  % passed in dzdy
  dzdy = varargin{1} ;
  varargin(1) = [] ;
else
  dzdy = [] ;
end

opts.lambda = 0.1 ;
opts.eta = 0.0005 ;
opts.mode = 'train' ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

orig_size = size(x) ;
x = x(:) ;

size(x)
size(v)

if opts.mode == 'train'
    if nargin <= 1 || isempty(dzdy)
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
    else
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
        v = v + opts.eta * (x * x') .* v ~= 0 ;
    end
else
    if nargin <= 1 || isempty(dzdy)
        y = x .* (ones(size(x)) + opts.lambda * v * x) ;
    else
        y = dzdy .* (ones(size(x)) + opts.lambda * (v * x + x .* sum(v, 2))) ;
    end
end

x = reshape(x, orig_size) ;
y = reshape(y, orig_size) ;