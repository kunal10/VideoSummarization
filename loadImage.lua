require 'math'
require 'image'

function loadim(imname)
    print(imname)
    im=image.load(imname);
    im=image.scale(im,224,224);
    if im:size(1)==1 then
        im2=torch.cat(im,im,1);
        im2=torch.cat(im2,im,1);
        im=im2;
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}];
    end
    im=im*255;
    im2=im:clone();
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68;
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779;
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939;
    return im2;
end

