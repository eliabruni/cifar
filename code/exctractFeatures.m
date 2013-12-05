opts.CIFAR_PATH='/Users/eliabruni/Downloads/cifar-100-matlab';
opts.ENCODER_PATH='/Users/eliabruni/work/2014/cifar/data/univBovwEncoder.mat';
opts.TRAIN_PATH='/Users/eliabruni/work/2014/cifar/data/trainBovwFeatures.mat';
opts.TEST_PATH='/Users/eliabruni/work/2014/cifar/data/testBovwFeatures.mat';


% --------------------------------------------------------------------
%                                            Compute training features
% --------------------------------------------------------------------

% load CIFAR training data
f1=load([opts.CIFAR_PATH '/train.mat']);
trx = [f1.data];
Ytrain = [f1.fine_labels] + 1;
clear f1;

Xtrain= zeros(32, 32, 3, 50000, 'uint8');

% load encoder
encoder = load(opts.ENCODER_PATH);

% extract bovw features
train = {};
for i=1:size(trx,1)
    im = imrotate(reshape(trx(i,:), 32, 32, 3), -90);   
    train{i} = encodeCifarImage(encoder,im);
end
train = cat(1, train{:});
train=rot90(train);
% save features
save(opts.TRAIN_PATH,'train');


% --------------------------------------------------------------------
%                                                Compute test features
% --------------------------------------------------------------------

% load CIFAR training data
f1=load([opts.CIFAR_PATH '/test.mat']);
trx = [f1.data];
Ytrain = [f1.fine_labels] + 1;
clear f1;

Xtrain= zeros(32, 32, 3, 10000, 'uint8');

% load encoder
encoder = load(opts.ENCODER_PATH);

% extract bovw features
test = {};
for i=1:size(trx,1)
    im = imrotate(reshape(trx(i,:), 32, 32, 3), -90);   
    test{i} = encodeCifarImage(encoder,im);
end
test = cat(1, test{:});
test=rot90(test);
% save features
save(opts.TEST_PATH,'test');