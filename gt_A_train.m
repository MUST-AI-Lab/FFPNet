close all
clear

gt_path = fullfile('part_A', 'train_data','ground-truth');
image_path = fullfile('part_A', 'train_data','images');
label_path = fullfile('part_A', 'train_data', 'labels_csr'); % 2, 4, 8
imgs_path = fullfile('part_A', 'train_data', 'images_csr');


if ~exist(gt_path, 'dir')
    gt_path
end
if ~exist(image_path, 'dir')
    image_path
end
gt = dir(fullfile(gt_path,'*.mat'));
images = dir(fullfile(image_path,'*.jpg'));
if ~exist(label_path, 'dir')
    mkdir(label_path);
end
if ~exist(imgs_path, 'dir')
    mkdir(imgs_path);
end

for idx=1:length(gt)
    disp([images(idx).name]);
    clear image_info;
    % read image
    im = imread(fullfile(image_path, images(idx).name));
    row = size(im, 1);
    col = size(im, 2);
    % load keypoints and transform coordinate
    load(fullfile(gt_path, gt(idx).name));
    points = image_info{1}.location;
    
    rects = zeros(9, 4);
    rects(1, :) = [1, 1, floor(col*0.5)-1, floor(row*0.5)-1];
    rects(2, :) = [1, floor(row*0.5)+1, floor(col*0.5)-1, row-floor(row*0.5)-1];
    rects(3, :) = [floor(col*0.5)+1, 1, col-floor(col*0.5)-1, floor(row*0.5)-1];
    rects(4, :) = [floor(col*0.5)+1, floor(row*0.5)+1, col-floor(col*0.5)-1, row-floor(row*0.5)-1];
    rects(5, :) = [floor(col*0.25)+1, floor(row*0.25)+1, floor(col*0.5)-1, floor(row*0.5)-1];
    rects(6, :) = [1, floor(row*0.25)+1, floor(col*0.5)-1, floor(row*0.5)-1];
    rects(7, :) = [floor(col*0.25)+1, 1, floor(col*0.5)-1, floor(row*0.5)-1];
    rects(8, :) = [floor(col*0.25)+1, floor(row*0.5)+1, floor(col*0.5)-1, row-floor(row*0.5)-1];
    rects(9, :) = [floor(col*0.5)+1, floor(row*0.25)+1, col-floor(col*0.5)-1, floor(row*0.5)-1];
    
    for i = 1:size(rects, 1)
        sub_img = imcrop(im ,rects(i, :));
        [sub_row, sub_col, sub_cnl] = size(sub_img);
        new_row = ceil(sub_row/16)*16;
        new_col = ceil(sub_col/16)*16;
        new_sub_img = imresize(sub_img, [new_row, new_col]);
        lt = rects(i, 1:2);
        rb = rects(i, 3:4) + lt;
        points_ = zeros(1, 2);
        num = 0;
        for k = 1:length(points)
            q_point = points(k, :);
            if q_point(1) >= lt(1) && q_point(2) >= lt(2) && q_point(1) <= rb(1) && q_point(2) <= rb(2)
                q_point(1) = q_point(1) - lt(1) + 1;
                q_point(2) = q_point(2) - lt(2) + 1;
                num = num + 1;
                points_(num, 1) = q_point(1);
                points_(num, 2) = q_point(2);
            end
        end
        points_(:,1) = ((points_(:,1)/sub_col-0.5)*(new_col/sub_col) + (new_col/sub_col)/2)*sub_col;
        points_(:,2) = ((points_(:,2)/sub_row-0.5)*(new_row/sub_row) + (new_row/sub_row)/2)*sub_row;

        mat_row = new_row / 8; % 2, 4, 8
        mat_col = new_col / 8; % 2, 4, 8
        sub_img_ = imresize(new_sub_img, [mat_row, mat_col]);
        Heat_Map = zeros(mat_row, mat_col);
        points_ = (points_) / 8; % 2, 4, 8
        
        for k = 1:size(points_, 1)
            query_point = points_(k,:);
            k_size = 15;
            Sigma = 4;
            current_Map = fspecial('gaussian', k_size, Sigma);
            x_ = max(1,floor(query_point(1)));
            y_ = max(1,floor(query_point(2)));
            radius = ceil(k_size/2);
            m = size(sub_img_, 1);
            n = size(sub_img_, 2);
            if (x_-radius+1<1)
                for ra = 0:radius-x_-1
                    current_Map(:,end-ra) = current_Map(:,end-ra)+current_Map(:,1);
                    current_Map(:,1)=[];
                end
            end  
            if (y_-radius+1<1)
               for ra = 0:radius-y_-1
                   current_Map(end-ra,:) = current_Map(end-ra,:)+current_Map(1,:);
                   current_Map(1,:)=[];
               end
            end
            if (x_+k_size-radius>n)   
               for ra = 0:x_+k_size-radius-n-1
                   current_Map(:,1+ra) = current_Map(:,1+ra)+current_Map(:,end);
                   current_Map(:,end) = [];
               end
            end
            if(y_+k_size-radius>m)    
                for ra = 0:y_+k_size-radius-m-1
                    current_Map(1+ra,:) = current_Map(1+ra,:)+current_Map(end,:);
                    current_Map(end,:) = [];
                end
            end
            Heat_Map(max(y_-radius+1,1):min(y_+k_size-radius,m),max(x_-radius+1,1):min(x_+k_size-radius,n))...
                = Heat_Map(max(y_-radius+1,1):min(y_+k_size-radius,m),max(x_-radius+1,1):min(x_+k_size-radius,n))...
                + current_Map;
        end
        if isnan(Sigma)
            continue;
        end
        Heat_Map = Heat_Map > 0;
        label_name = fullfile(label_path, strcat(gt(idx).name(4:end-4), num2str(i), '.mat'));
        img_name = fullfile(imgs_path, strcat(gt(idx).name(4:end-4), num2str(i), '.jpg'));
        save(label_name, 'Heat_Map');
        % imwrite(new_sub_img, img_name);
        % horizontal flipping
        label_name_f = fullfile(label_path, strcat(gt(idx).name(4:end-4), num2str(i), '_f.mat'));
        img_name_f = fullfile(imgs_path, strcat(gt(idx).name(4:end-4), num2str(i), '_f.jpg'));
        Heat_Map = fliplr(Heat_Map);
        new_sub_img_f = fliplr(new_sub_img);
        save(label_name_f, 'Heat_Map');
        % imwrite(new_sub_img_f, img_name_f);
        % imagesc(Heat_Map);
        % imshow(sub_img);
    end
    
end
disp('Done!');