function descrs = encodeCifarImage(encoder, im)



descrs = encodeOne(encoder, im) ;



% --------------------------------------------------------------------
function psi = encodeOne(encoder, im)
% --------------------------------------------------------------------

features = encoder.extractorFn(im) ;

imageSize = size(im) ;
psi = {} ;
for i = 1:size(encoder.subdivisions,2)
    minx = encoder.subdivisions(1,i) * imageSize(2) ;
    miny = encoder.subdivisions(2,i) * imageSize(1) ;
    maxx = encoder.subdivisions(3,i) * imageSize(2) ;
    maxy = encoder.subdivisions(4,i) * imageSize(1) ;
    
    ok = ...
        minx <= features.frame(1,:) & features.frame(1,:) < maxx  & ...
        miny <= features.frame(2,:) & features.frame(2,:) < maxy ;
    
    descrs = encoder.projection * bsxfun(@minus, ...
        features.descr(:,ok), ...
        encoder.projectionCenter) ;
    if encoder.renormalize
        descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
    end
    
    w = size(im,2) ;
    h = size(im,1) ;
    frames = features.frame(1:2,:) ;
    frames = bsxfun(@times, bsxfun(@minus, frames, [w;h]/2), 1./[w;h]) ;
    
    descrs = extendDescriptorsWithGeometry(encoder.geometricExtension, frames, descrs) ;
    
    switch encoder.type
        case 'bovw'
            [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                descrs, ...
                'MaxComparisons', 100) ;
            z = vl_binsum(zeros(encoder.numWords,1), 1, double(words)) ;
            z = sqrt(z) ;
            
        case 'fv'
            z = vl_fisher(descrs, ...
                encoder.means, ...
                encoder.covariances, ...
                encoder.priors, ...
                'Improved') ;
        case 'vlad'
            [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                descrs, ...
                'MaxComparisons', 15) ;
            assign = zeros(encoder.numWords, numel(words), 'single') ;
            assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
            z = vl_vlad(descrs, ...
                encoder.words, ...
                assign, ...
                'SquareRoot', ...
                'NormalizeComponents') ;
    end
    z = z / max(sqrt(sum(z.^2)), 1e-12) ;
    psi{i} = z(:) ;
end
psi = cat(1, psi{:}) ;