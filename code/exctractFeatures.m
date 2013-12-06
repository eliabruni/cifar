opts.CIFAR_PATH='';

% if true it reuses previously computed and saved data
opts.reuseSavedData = true;
data.prefix = 'bovw';
data.dir = 'data';

for pass = 1:2
    data.resultDir = fullfile(data.dir, data.prefix);
    data.encoderPath = fullfile(data.resultDir, 'encoder.mat');
    data.trainFeaturesPath =  fullfile(data.resultDir, 'trainBovwFeatures.mat');
    data.trainFeaturesPath =  fullfile(data.resultDir, 'testBovwFeatures.mat');
end

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
    'step', 4, ...
    'scales', 2.^(1:-.5:-3))};



% load CIFAR training data
f1=load([opts.CIFAR_PATH '/train.mat']);
trx = [f1.data];
clear f1;

% --------------------------------------------------------------------
%                                                        Train encoder
% --------------------------------------------------------------------

if exist(data.encoderPath) & opts.reuseSavedData
    encoder = load(data.encoderPath);
else
    encoder = trainEncoder(trx, ...
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
save(data.trainFeaturesPath,'train');


% --------------------------------------------------------------------
%                                                Compute test features
% --------------------------------------------------------------------

% load CIFAR training data
f1=load([opts.CIFAR_PATH '/test.mat']);
trx = [f1.data];
clear f1;

% extract bovw features
test = {};
for i=1:size(trx,1)
    im = imrotate(reshape(trx(i,:), 32, 32, 3), -90);
    test{i} = encodeCifarImage(encoder,im);
end
test = cat(2, test{:});
test=rot90(test);
% save features
save(data.trainFeaturesPath,'test');