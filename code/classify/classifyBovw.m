
CIFAR_DIR='/Users/eliabruni/work/2014/cifar/data/cifar-100-matlab';

assert(~strcmp(CIFAR_DIR, '/path/to/cifar/cifar-10-batches-mat/'), ...
    ['You need to modify kmeans_demo.m so that CIFAR_DIR points to ' ...
    'your cifar-10-batches-mat directory.  You can download this ' ...
    'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);

% if true it reuses previously computed and saved data
opts.reuseSavedData = true;
%data.prefix = 'bovw';
%data.dir = 'data';

addpath('/Users/eliabruni/git/cifar/code');

for pass = 1:2
    %data.resultDir = fullfile(data.dir, data.prefix);
    data.encoderPath = '/Users/eliabruni/work/2014/cifar/data/bovw/encoder.mat';
    %data.trainFeaturesPath =  fullfile(data.resultDir, 'trainBovwFeatures.mat');
    %data.trainFeaturesPath =  fullfile(data.resultDir, 'testBovwFeatures.mat');
end

%% Configuration
CIFAR_DIM=[32 32 3];

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/train.mat']);

trainX = double(f1.data);
trainY = double(f1.fine_labels) + 1; % add 1 to labels!
clear f1;

% feature extraction and encoding parameters
opts.encoderParams = {...
    'maxNumTrainImages', 10000, ...
    'type', 'bovw', ...
    'numWords', 4096, ...
    'layouts', {'1x1'}, ...
    'geometricExtension', 'xy', ...
    'numPcaDimensions', 100, ...
    'whitening', true, ...
    'whiteningRegul', 0.01, ...
    'renormalize', true, ...
    'extractorFn', @(x) getDenseSIFT(x, ...
    'step', 1, ...
    'scales', 2.^(1:-.5:-3))};

% --------------------------------------------------------------------
%                                                        Train encoder
% --------------------------------------------------------------------

if exist(data.encoderPath) & opts.reuseSavedData
    encoder = load(data.encoderPath);
else
    encoder = cifarTrainEncoder(trx, ...
        opts.encoderParams{:});
    save(data.encoderPath, '-struct', 'encoder');
    fprintf('Traning encoder done!\n\n');
    diary off;
    diary on;
end


% --------------------------------------------------------------------
%                                            Compute training features
% --------------------------------------------------------------------

% extract bovw features
train = {};
for i=1:size(trx,1)
    im = imrotate(reshape(trx(i,:), 32, 32, 3), -90);
    train{i} = encodeCifarImage(encoder,im);
end
train = cat(2, train{:});
train=rot90(train);
% save features
%save(data.trainFeaturesPath,'train');


% train classifier using SVM
C = 100;
theta = train_svm(train, trainY, C);

[val,labels] = max(train*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

% %% Load CIFAR test data
% fprintf('Loading test data...\n');
% f1=load([CIFAR_DIR '/test.mat']);
% testX = double(f1.data);
% testY = double(f1.fine_labels) + 1;
% clear f1;
% 
% % compute testing features and standardize
% if (whitening)
%     testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
% else
%     testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
% end
% testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
% testXCs = [testXCs, ones(size(testXCs,1),1)];
% 
% % test and print result
% [val,labels] = max(testXCs*theta, [], 2);
% fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

