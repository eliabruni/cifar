function encoder = cifarTrainEncoder(trx, varargin)


opts.type = 'bovw';
opts.numWords = [];
opts.seed = 1;
opts.numPcaDimensions = +inf;
opts.whitening = false;
opts.whiteningRegul = 0;
opts.numSamplesPerWord = [];
opts.renormalize = false;
opts.layouts = {'1x1'};
opts.geometricExtension = 'none';
opts.subdivisions = zeros(4,0);
opts.readImageFn = @readImage;
opts.extractorFn = @getDenseSIFT;
opts.lite = false;
opts = vl_argparse(opts, varargin);


for i = 1:numel(opts.layouts)
    t = sscanf(opts.layouts{i},'%dx%d');
    m = t(1);
    n = t(2);
    [x,y] = meshgrid(...
        linspace(0,1,n+1), ...
        linspace(0,1,m+1));
    x1 = x(1:end-1,1:end-1);
    y1 = y(1:end-1,1:end-1);
    x2 = x(2:end,2:end);
    y2 = y(2:end,2:end);
    opts.subdivisions = cat(2, opts.subdivisions, ...
        [x1(:)';
        y1(:)';
        x2(:)';
        y2(:)'] );
end

if isempty(opts.numWords)
    switch opts.type
        case {'bovw'}
            opts.numWords = 1024;
        case {'fv'}
            opts.numWords = 64;
            opts.numPcaDimensions = 80;
        case {'vlad'}
            opts.numWords = 64;
            opts.numPcaDimensions = 100;
            opts.whitening = true;
            opts.whiteninRegul = 0.01;
        otherwise
            assert(false);
    end
end

if isempty(opts.numSamplesPerWord)
    switch opts.type
        case {'bovw'}
            opts.numSamplesPerWord = 200;
        case {'vlad','fv'}
            opts.numSamplesPerWord = 1000;
        otherwise
            assert(false);
    end
    if opts.lite
        opts.numSamplesPerWord = 10;
    end
end

disp('Encoder options:' ); disp(opts);

encoder.type = opts.type;
encoder.subdivisions = opts.subdivisions;
encoder.readImageFn = opts.readImageFn;
encoder.extractorFn = opts.extractorFn;
encoder.numWords = opts.numWords;
encoder.renormalize = opts.renormalize;
encoder.geometricExtension = opts.geometricExtension;

%% Step 0: obtain sample image descriptors
numImages = size(trx,1);
numDescrsPerImage = ceil(opts.numWords * opts.numSamplesPerWord / numImages);
descrs = cell(1,numImages);
frames = cell(1,numImages);
parfor i=1:size(trx,1)
    try
        im = imrotate(reshape(trx(i,:), 32, 32, 3), -90);
        w = size(im,2);
        h = size(im,1);
        features = encoder.extractorFn(im);
        randn('state',0);
        rand('state',0);
        sel = vl_colsubset(1:size(features.descr,2), single(numDescrsPerImage));
        descrs{i} = features.descr(:,sel);
        frames{i} = features.frame(:,sel);
        frames{i} = bsxfun(@times, bsxfun(@minus, frames{i}(1:2,:), [w;h]/2), 1./[w;h]);
    catch
        fprintf(1, 'Error reading file: %d\n', i);
    end
end
descrs = cat(2, descrs{:});
frames = cat(2, frames{:});

%% Step 1 (optional): learn PCA projection
if opts.numPcaDimensions < inf || opts.whitening
    fprintf('%s: learning PCA rotation/projection\n', mfilename);
    encoder.projectionCenter = mean(descrs,2);
    x = bsxfun(@minus, descrs, encoder.projectionCenter);
    X = x*x' / size(x,2);
    [V,D] = eig(X);
    d = diag(D);
    [d,perm] = sort(d,'descend');
    d = d + opts.whiteningRegul * max(d);
    m = min(opts.numPcaDimensions, size(descrs,1));
    V = V(:,perm);
    if opts.whitening
        encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)';
    else
        encoder.projection = V(:,1:m)';
    end
    clear X V D d;
else
    encoder.projection = 1;
    encoder.projectionCenter = 0;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter);
if encoder.renormalize
    descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2))));
end


%% Step 2 (optional): geometrically augment the features

descrs = extendDescriptorsWithGeometry(opts.geometricExtension, frames, descrs);

%% Step 3: learn a VQ or GMM vocabulary
dimension = size(descrs,1);
numDescriptors = size(descrs,2);

switch encoder.type
    case {'bovw', 'vlad'}
        vl_twister('state', opts.seed);
        encoder.words = vl_kmeans(descrs, opts.numWords, 'verbose', 'algorithm', 'elkan');
        encoder.kdtree = vl_kdtreebuild(encoder.words, 'numTrees', 2);
        
    case {'fv'};
        vl_twister('state', opts.seed);
        if 1
            v = var(descrs')';
            [encoder.means, encoder.covariances, encoder.priors] = ...
                vl_gmm(descrs, opts.numWords, 'verbose', ...
                'Initialization', 'kmeans', ...
                'CovarianceBound', double(max(v)*0.0001), ...
                'NumRepetitions', 1);
        else
            addpath lib/yael/matlab
            [a,b,c] = ...
                yael_gmm(descrs, opts.numWords, 'verbose', 2);
            encoder.priors = single(a);
            encoder.means = single(b);
            encoder.covariances = single(c);
        end
end
