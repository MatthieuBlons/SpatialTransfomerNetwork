# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:44:05 2021

@author: MBlons
"""
import numpy as np
import tensorflow as tf


#%% ------------------------------- 2D ----------------------------------------
#%% Apply 2D tform to 2D Img 
class Apply2DTform(tf.keras.layers.Layer):
    
    """
    Spatial Transformer layer implementation as described in [1].
    
    The layer is composed of 4 elements:
        
    -   _Imwarp: takes the input 2D image Batch <Img> of size (B, H, W, 1), 
        the transformation vector <Tform> of size (B, 6), the output size 
        <output_size> of a sample Img in the batch (H*, W*, 1), the padding 
        method <padding> and, the interpolation method <interp> 
        and outputs the warped output Batch of size (B, H*, W*, 1)
    
    -   _meshgrid: generates a grid of (x, y) coordinates, wth regards to the 
        transformation, that correspond to a set of points where the 
        input should be sampled to produce the transformed output.
      
    -   _interpolate: takes as input the original image, the grid (x, y)
        and, and produces the output transformed image using the intrepolation 
        method
      
    -   _get_pixel_value: takes an image as input and outputs pixels value at 
        (x, y) coordinates.
      
      
    Input
    -----
    -   input_size<tuple>: size of the input image (H*, W*, 1)
    -   output_size<tuple>: size of the output image (H*, W*, 1)
    -   Img<tensor>: the input 2D image batch of size (B, H, W, 1)
    -   Tform<tensor>: transformations to apply to each input sample (B, 6)
        Initialize to identity matrix.
    -   padding<bool>: apply padding before interpolation True/False
    -   interp<str>: interpolation method; supported_interp = ['Bilinear','Nearest']
    
    Returns
    -------
    -   out: transformed input image batch. Tensor of size (B, H*, W*, 1).
    
    Use
    ---
    
    out = Apply2DTform(input_size, output_size)(Img, Tform, padding, interp)
    
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    
    def __init__(self, input_size, output_size, **kwargs):
        self.output_size = output_size
        self.input_size = input_size
        super(Apply3DTform, self).__init__(**kwargs)
        
    def get_config(self):
        return {
                'output_size' : self.output_size, 
        }
    
    def compute_output_shape(self, input_shape):
        self.height, self.width, self.depth = self.output_size
        channels = input_shape[-1]
        return (None, self.height, self.width, self.depth, channels)
        
    def call(self, Img, Tform, padding=False, interp='Bilinear'):
    
        supported_interp = ['Bilinear','Nearest']
        try:
            test = interp in supported_interp
            if test == False:
                raise ValueError('Wrong Keyword For Interp Method')
        except ValueError:
            print("Supported interp keywords : 'Bilinear' (Default) or 'Nearest'")
       
        output = self._Imwarp(Img, Tform, self.input_size, self.output_size, padding, interp)
        return output
    
    def _interpolate(self, Img, x_s, y_s, method):
        pad = [[0,0],
               [0,1], 
               [0,1], 
               [0,0]]
        
        Img = tf.pad(Img, pad)
        
        # grab input dimension  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        
        #
        max_x = tf.cast(H , 'int32')-1
        max_y = tf.cast(W , 'int32')-1
      
        # resacle x_s, y_s to [0, W-1], [0, H-1], [0, D-1]
        x = tf.cast(x_s, 'float32')
        y = tf.cast(y_s, 'float32')

        x = .5*(x + 1)*tf.cast(max_x-1, 'float32')
        y = .5*(y + 1)*tf.cast(max_y-1, 'float32')

        # grab corners points for each (x_i, y_i, z_i)
        x0 = tf.cast(tf.round(x), 'int32')        
        x1 = x0 + 1
        y0 = tf.cast(tf.round(y), 'int32')
        y1 = y0 + 1

        #clip to range [0, W-1], [0, H-1], [0, D-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x) 
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y) 
          
        # get pixel value at corner coords
        
        I00 = self._get_pixel_value(Img, x0, y0)
        I01 = self._get_pixel_value(Img, x0, y1)
        I10 = self._get_pixel_value(Img, x1, y0)
        I11 = self._get_pixel_value(Img, x1, y1)

        # recast as float to calculation
        
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')   

        # claculate deltas v
        
        if method == 'Bilinear':
            W00 = (x1-x)*(y1-y)
            W11 = (x-x0)*(y-y0)
            W01 = (x1-x)*(y-y0)
            W10 = (x-x0)*(y1-y)
            
        elif method == 'Nearest':
            W00 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))
            W11 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))
            W01 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))
            W10 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))
                   
        #add dim
        W00 = tf.expand_dims(W00, axis = -1)
        W11 = tf.expand_dims(W11, axis = -1)
        W01 = tf.expand_dims(W01, axis = -1)
        W10 = tf.expand_dims(W10, axis = -1)
        
        out = tf.add_n([W00*I00, W01*I01, W10*I10, W11*I11])
        
        return out  
                
    def _meshgrid(self, height, width, vector, matrix):
        # grab nb batchs  
        batchs = tf.shape(vector)[0]
        # def grid
        ax, ay = tf.linspace(-1., 1., height), tf.linspace(-1., 1., width)
        x_t, y_t = tf.meshgrid(ax, ay)
        
        # x_t = tf.transpose(x_t)
        # y_t = tf.transpose(y_t)
        
        x_t = tf.keras.backend.permute_dimensions(x_t, (1,0))
        y_t = tf.keras.backend.permute_dimensions(y_t, (1,0))
        
        # flaten and reshape to [x_t, y_t] (non homogeneous form)
        x_t_flat, y_t_flat = tf.reshape(x_t, [-1]), tf.reshape(y_t, [-1])
        sampling_grid = tf.stack([x_t_flat, y_t_flat])  
        # repeat sampling grid over batch    
        sampling_grid = tf.expand_dims(sampling_grid, axis = 0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([batchs, 1, 1]))
        # cast tform to float for linalg ops
        # Translation
        vector = tf.expand_dims(vector, axis = -1)
        vector = tf.tile(vector, tf.stack([1, 1, height*width]))
        vector = tf.cast(vector, 'float32')  
        # Rot/Scale/Shear
        matrix = tf.cast(matrix, 'float32')
        # Grid
        sampling_grid = tf.cast(sampling_grid, 'float32')
        # batch multiply tform sample grid
        # batchs_grids = tf.linalg.matmul(tf.linalg.inv(matrix), tf.add(sampling_grid, -vector))  # shape(B, 3, H*W)
        batchs_grids = tf.add(tf.linalg.matmul(matrix, sampling_grid), vector) # shape(B, 3, H*W)
        # reshape to (B, 2, H, W)
        batchs_grids = tf.reshape(batchs_grids[:, :, :], [batchs, 2, height, width])
        # return T grid
        return batchs_grids
        
    def _get_pixel_value(self, Img, x, y):
        # grab grid dim       
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        # enumerate Img in batch
        batch_indx = tf.range(0, B)
        # reshape batch indx
        batch_indx = tf.reshape(batch_indx, (B, 1, 1))
        batch_indx = tf.tile(batch_indx, (1, H, W))
        # stack coordinates
        indices = tf.stack([batch_indx, x, y], axis=-1)
        # return intensity at (x, y)
        return tf.gather_nd(Img, indices)  
    
    def _Imwarp(self, Img, Tform, input_size, output_size, padding, interp):        
        # grab Img size  
        B = tf.shape(Img)[0]
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        # generate Id 
        # Id = tf.tile(tf.expand_dims(tf.eye(2), axis=0), [B,1,1])
        # Grab and Reshape Translation
        V = tf.reshape(Tform[:, 4:], [B, 2]) 
        V = tf.cast(V, 'float32')
        # Grab and Reshape Rot/Scale/Shear
        M = tf.reshape(Tform[:, :4], [B, 2, 2]) 
        M = tf.cast(M, 'float32')
        # generate grids
        indx_grid = self._meshgrid(height = output_size[0], width = output_size[1], vector=V, matrix=M)
        x_s = indx_grid[:,0,:,:]
        y_s = indx_grid[:,1,:,:]
        # padding if requested
        if padding:
            ph = (output_size[0]-H)/2
            pw = (output_size[1]-W)/2
            pad = [[0,0],
                   [np.ceil(ph).astype('int'), np.floor(ph).astype('int')],
                   [np.ceil(pw).astype('int'), np.floor(pw).astype('int')],
                   [0,0]]          
            Img = tf.pad(Img, pad)
            
        # padding if requested
        if padding:
            H = input_size[0]
            W = input_size[1]

            top_pad = np.int32(np.ceil((output_size[0]-H)/2))
            bottom_pad = np.int32(np.floor((output_size[0]-H)/2))
            left_pad = np.int32(np.ceil((output_size[1]-W)/2))
            right_pad = np.int32(np.floor((output_size[1]-W)/2))

            pad = ((top_pad, bottom_pad), (left_pad, right_pad))      
            
            Img = tf.keras.layers.ZeroPadding2D(padding=pad)(Img) 
                
        #sample input with grid to get output    
        output_Img = self._interpolate(Img, x_s, y_s, method=interp)

        return output_Img  

#%% Apply 2D Displacement field to 2D image 
class Apply2DDispField(tf.keras.layers.Layer):
    
    """
    Spatial Transformer layer implementation for 2D displacement field
    
    The layer is composed of 4 elements:
        
    -   _Imwarp: takes the input 2D image Batch <Img> of size (B, H, W, 1), 
        the displacement field <DispField> of size (B, H*, W*, 2), the output size 
        <output_size> of a sample Img in the batch (H*, W*, 1), the padding 
        method <padding> and, the interpolation method <interp> 
        and outputs the warped output Batch of size (B, H*, W*, 1)
    
    -   _meshgrid: generates a grid of (x, y) coordinates, wth regards to the 
        transformation, that correspond to a set of points where the 
        input should be sampled to produce the transformed output.
      
    -   _interpolate: takes as input the original image, the grid (x, y)
        and, and produces the output transformed image using the intrepolation 
        method
      
    -   _get_pixel_value: takes an image as input and outputs pixels value at 
        (x, y) coordinates.
      
      
    Input
    -----
    -   input_size<tuple>: size of the input image (H*, W*, 1)
    -   output_size<tuple>: size of the output image (H*, W*, 1)
    -   Img<tensor>: the input 3D image batch of size (B, H, W, D, 1)
    -   DispField<tensor>: displacement field to apply to each input sample (B, H*, W*, 2)
        Initialize to null.
    -   padding<bool>: apply padding before interpolation True/False
    -   interp<str>: interpolation method; supported_interp = ['Bilinear','Nearest']
    
    Returns
    -------
    -   out: transformed input image batch. Tensor of size (B, H*, W*, 1).
    
    Use
    ---
    
    out = Apply2DDispField(input_size, output_size)(Img, DispField, padding, interp)
    
    """
    
    def __init__(self, input_size, output_size, **kwargs):
        self.output_size = output_size
        self.input_size = input_size
        super(Apply2DDispField, self).__init__(**kwargs)
        
    def get_config(self):
        return {
                'output_size' : self.output_size, 
        }
    
    def compute_output_shape(self, input_shape):
        self.height, self.width = self.output_size
        channels = input_shape[-1]        
        return (None, self.height, self.width, channels)
    
    def call(self, Img, DispField, padding=False, interp='Bilinear'):
        supported_interp = ['Bilinear','Nearest']
        try:
            test = interp in supported_interp
            if test == False:
                raise ValueError('Wrong Keyword For Interp Method')
        except ValueError:
            print("Supported interp keywords : 'Bilinear' (Default) or 'Nearest'")     
        output = self._Imwarp(Img, DispField, self.input_size, self.output_size, padding, interp)        
        return output
    
    def _meshgrid(self, height, width, displacement):
        # grab nb batchs  
        batchs = tf.shape(displacement)[0]
        # def grid
        ax, ay = tf.linspace(-1., 1., height), tf.linspace(-1., 1., width)
        x_t, y_t = tf.meshgrid(ax,ay)
        # x_t, y_t = tf.transpose(x_t), tf.transpose(y_t)
        x_t = tf.keras.backend.permute_dimensions(x_t, (1,0))
        y_t = tf.keras.backend.permute_dimensions(y_t, (1,0))
        # flaten and reshape to [x_t, y_t] (non homogeneous form)
        x_t_flat, y_t_flat = tf.reshape(x_t, [-1]), tf.reshape(y_t, [-1])
        sampling_grid = tf.stack([x_t_flat, y_t_flat])  
        # repeat sampling grid over batch    
        sampling_grid = tf.expand_dims(sampling_grid, axis = 0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([batchs, 1, 1]))
        # reshape RegField
        indx_grid = tf.keras.backend.permute_dimensions(displacement, pattern=(0,3,1,2))
        indx_grid = tf.cast(indx_grid, 'float32')        
        dx = indx_grid[:,0,:,:]
        dy = indx_grid[:,1,:,:]
        dx_flat, dy_flat = tf.reshape(dx, [batchs, height*width]), tf.reshape(dy, [batchs, height*width])
        d_grid = tf.stack([dx_flat, dy_flat], axis=1)          
        # Apply Displacement 
        batchs_grids = sampling_grid - d_grid
        #reshape to (B, 2, H, W, D)
        batchs_grids = tf.reshape(batchs_grids, [batchs, 2, height, width])
        # return T grid
           
        return batchs_grids
    
    def _interpolate(self, Img, x_s, y_s, method):
        # Padding for last row/col
        pad = [[0,0],
               [0,1], 
               [0,1], 
               [0,0]]
        Img = tf.pad(Img, pad)
        # grab input dimension  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        # Def max indx
        max_x = tf.cast(H , 'int32')-1
        max_y = tf.cast(W , 'int32')-1
        # resacle x_s, y_s, z_s to [0, W-2], [0, H-2], [0, D-2]
        x = tf.cast(x_s, 'float32')
        y = tf.cast(y_s, 'float32')
        x = .5*(x + 1)*tf.cast(max_x-1, 'float32')
        y = .5*(y + 1)*tf.cast(max_y-1, 'float32')
        # grab corners points for each (x_i, y_i, z_i)
        x0 = tf.cast(tf.round(x), 'int32')        
        x1 = x0 + 1
        y0 = tf.cast(tf.round(y), 'int32')
        y1 = y0 + 1
        #clip to range [0, W-1], [0, H-1], [0, D-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x) 
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y) 
        # get pixel value at corner coords
        I00 = self._get_pixel_value(Img, x0, y0)
        I01 = self._get_pixel_value(Img, x0, y1)
        I10 = self._get_pixel_value(Img, x1, y0)
        I11 = self._get_pixel_value(Img, x1, y1)
        # recast as float for calculation       
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')   
        # claculate deltas v
        # Bilinear interp
        if method == 'Bilinear':
            W00 = (x1-x)*(y1-y)
            W11 = (x-x0)*(y-y0)
            W01 = (x1-x)*(y-y0)
            W10 = (x-x0)*(y1-y)
        # Nearest neighbour
        elif method == 'Nearest':
            W00 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))
            W11 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))
            W01 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))
            W10 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))
        # add dim
        W00 = tf.expand_dims(W00, axis = -1)
        W11 = tf.expand_dims(W11, axis = -1)
        W01 = tf.expand_dims(W01, axis = -1)
        W10 = tf.expand_dims(W10, axis = -1)
        # return interp Output 
        out = tf.add_n([W00*I00, W01*I01, W10*I10, W11*I11])        
        
        return out   
        
    def _get_pixel_value(self, Img, x, y):
        # grab grid dim       
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        # enumerate Img in batch
        batch_indx = tf.range(0, B)
        # reshape batch indx
        batch_indx = tf.reshape(batch_indx, (B, 1, 1))
        batch_indx = tf.tile(batch_indx, (1, H, W))
        # stack coordinates
        indices = tf.stack([batch_indx, x, y], 3)
        # return intensity at (x, y)
        
        return tf.gather_nd(Img, indices)  
        
    def _Imwarp(self, Img, DispField, input_size, output_size, padding, interp):        
        # grab Img size  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        # generate grids
        indx_grid = self._meshgrid(height = output_size[0], width = output_size[1], displacement = DispField)
        x_s = indx_grid[:,0,:,:]
        y_s = indx_grid[:,1,:,:]
        # padding if requested
        if padding:
            H = input_size[0]
            W = input_size[1]
        
            top_pad = np.int32(np.ceil((output_size[0]-H)/2))
            bottom_pad = np.int32(np.floor((output_size[0]-H)/2))
            left_pad = np.int32(np.ceil((output_size[1]-W)/2))
            right_pad = np.int32(np.floor((output_size[1]-W)/2))
        
            pad = ((top_pad, bottom_pad), (left_pad, right_pad))      
            
            Img = tf.keras.layers.ZeroPadding2D(padding=pad)(Img) 
            
        #sample input with grid to get output    
        output_Img = self._interpolate(Img, x_s, y_s, method=interp)

        return output_Img  
    
#%% Generate a batch of random affine tform matrices for 2D images
def _2Dtform_generation(batch=1, dim=(28,28), limit_t=[14, 14], limit_r=90, limit_scale=0.1, limit_shear=0.1, distribution='uniform'):
    """
    _2Dtform_generation: generate a minibatch of random 2D tranformation matrices

    Input
    -----
    -   batch<int>: number of tforms to generate 
    -   dim<tuple>: Dimension of the image to transform 
    -   limit_t<list of int>: Maximum of amplitude for the translation (in voxel)
    -   limit_r<list of float>: Maximum of amplitude for the rotation (in °)
    -   limit_scale<float>: Maximum of amplitude for the scaling
    -   limit_shear<float>: Maximum of amplitude for the shear
    -   distribution<str>: Distribution from which to draw tform params. 'uniform' or 'normal' are supported.
    
    Returns
    -------
    -   tform: tform batch. Tensor of size (B, 6).
    -   [tx, ty]: list of translation for each tform.
    -   [rz]: list of rotation for each tform.
    -   [scx, scy]: list of scale for each tform.
    -   [shx, shy]: list of scale for each tform.
    
    Use
    ---
    
    tform, [tx, ty], [rz], [scx, scy], [shx, shy] = _2Dtform_generation(batch = 10, 
                                                                          dim  (14,14), 
                                                                          limit_t = [14,14],  
                                                                          limit_r = 90, 
                                                                          limit_scale = 0, 
                                                                          limit_shear = 0,
                                                                          distribution = 'uniform')
    
    """
    supported_dist = ['uniform', 'normal']
    try:
        test = distribution in supported_dist
        if test == False:
            raise ValueError('Wrong Keyword argument For distribution ')
    except ValueError:
        print("Supported distribution : 'uniform','normal'")
    
    # tform Parameters Distribution 
    if distribution == 'uniform':
        # T             
        tx = tf.random.uniform([batch, 1], -limit_t[0], limit_t[0])/dim[0]*2
        ty = tf.random.uniform([batch, 1], -limit_t[1], limit_t[1])/dim[1]*2     
        # R                   
        rz = tf.random.uniform([batch, 1], -limit_r, limit_r)*(np.pi/180) #deg to rad
        # Scale
        scx = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
        scy = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
        # Shear
        shx = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
        shy = tf.random.uniform([batch, 1], -limit_shear, limit_shear)        
    elif distribution == 'normal':
        # T             
        tx = tf.random.normal([batch, 1], 0, (limit_t[0]/dim[0]*2)/3)
        ty = tf.random.normal([batch, 1], 0, (limit_t[1]/dim[1]*2)/3)    
        # R                   
        rz = tf.random.normal([batch, 1], 0, (limit_r*(np.pi/180))/3) #deg to rad
        # Scale
        scx = tf.random.normal([batch, 1], 0, limit_scale/3)
        scy = tf.random.normal([batch, 1], 0, limit_scale/3)
        # Shear
        shx = tf.random.normal([batch, 1], 0, limit_shear/3)
        shy = tf.random.normal([batch, 1], 0, limit_shear/3)
            
    # Translation              
    trans = tf.concat([tx, ty], axis=-1)
        
    # Rotation
    cos = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(rz)
    sin = tf.tile(tf.cast(tf.expand_dims([0,-1,1,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(rz)
    rot = cos+sin

    # Scaling
    scalex = tf.tile(tf.cast(tf.expand_dims([1,0,0,0], axis=0), dtype='float32'), [batch, 1])*scx
    scaley = tf.tile(tf.cast(tf.expand_dims([0,0,0,1], axis=0), dtype='float32'), [batch, 1])*scy
    scale = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])+scalex+scaley

    # Shear
    shearx = tf.tile(tf.cast(tf.expand_dims([0,1,0,0], axis=0), dtype='float32'), [batch, 1])*shx
    sheary = tf.tile(tf.cast(tf.expand_dims([0,0,1,0], axis=0), dtype='float32'), [batch, 1])*shy
    shear = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])+shearx+sheary
    
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(shear, [batch, 2, 2]), tf.linalg.matmul(tf.reshape(scale, [batch, 2, 2]), tf.reshape(rot, [batch, 2, 2])))

    tform = tf.concat([tf.reshape(matrix, [batch, 4]), trans], axis=-1)
    
    return tform, [tx, ty], [rz], [scx, scy], [shx, shy]

#%% Generate a single 2D tform matrix from input params
def Make_2Dtform(dim = [32,32], trans = [16,16], rot = 0, scale = [0, 0], shear=[0, 0]):
    """
    Make_2Dtform: generates a 2D tranformation matrix with specific params
    
    Input
    -----
    -   dim<tuple>: Dimension of the volume to transform 
    -   trans<list of int>: translation (in voxel)
    -   rot<lfloat>: rotation (in °)
    -   scale<list of float>: scaling
    -   shear<list of float>: shear

    
    Returns
    -------
    -   tform: Tensor of size (1, 6).
    
    Use
    ---
    
    tform = Make_3Dtform(dim = (32,32), 
                         trans = [16,16],  
                         rot = 0, 
                         scale = [0, 0], 
                         shear = [0, 0])
    
    """
    # Translation              
    trans = tf.cast(trans, 'float32')/tf.cast(dim, 'float32')*2
    tx = trans[0]    
    ty = trans[1] 
    
    # Rotation                   
    rz = rot*(np.pi/180) #deg to rad 
    
    cos = tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32')*tf.math.cos(rz)
    sin = tf.cast(tf.expand_dims([0,-1,1,0], axis=0), dtype='float32')*tf.math.sin(rz)
    
    rot = cos+sin

    # Scaling
    scx = scale[0]
    scy = scale[1]

    scalex = tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32')*scx
    scaley = tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32')*scy

    scale = tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32')+scalex+scaley
           
    # Shear
    shx = shear[0]
    shy = shear[1]
    
    shearx = tf.cast(tf.expand_dims([0,1,0,0], axis=0), dtype='float32')*shx
    sheary = tf.cast(tf.expand_dims([0,0,1,0], axis=0), dtype='float32')*shy
    
    shear = tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32')+shearx+sheary
     
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(shear, [2, 2]), tf.linalg.matmul(tf.reshape(scale, [2, 2]), tf.reshape(rot, [2, 2])))

    tform = tf.concat([tf.reshape(matrix, [1, 4]), trans], axis=-1)
    
    return tform, [tx, ty], [rz], [scx, scy], [shx, shy] 
    
#%% ------------------------------- 3D ----------------------------------------
#%% Apply 3D tform to 3D Img
class Apply3DTform(tf.keras.layers.Layer):
    """
    Spatial Transformer layer implementation as described in [1]. Adapted for 
    3D images 
    
    Methods
    -----

    -   _Imwarp: takes the input 3D image Batch <Img> of size (B, H, W, D, 1), 
        the transformation vector <Tform> of size (B, 12), the output size 
        <output_size> of a sample Img in the batch (H*, W*, D*, 1), the padding 
        method <padding> and, the interpolation method <interp> 
        and outputs the warped output Batch of size (B, H*, W*, D*, 1)
    
    -   _meshgrid: generates a grid of (x, y, z) coordinates, wth regards to the 
        transformation, that correspond to a set of points where the 
        input should be sampled to produce the transformed output.
      
    -   _interpolate: takes as input the original image, the grid (x, y, z)
        and, and produces the output transformed image using the intrepolation 
        method
      
    -   _get_pixel_value: takes an image as input and outputs pixels value at 
        (x, y, z) coordinates.

    Input
    -----
    -   input_size<tuple>: size of the input image (H*, W*, D*, 1)
    -   output_size<tuple>: size of the output image (H*, W*, D*, 1)
    -   Img<tensor>: the input 3D image batch of size (B, H, W, D, 1)
    -   Tform<tensor>: transformations to apply to each input sample (B, 12)
    -   padding<bool>: apply padding before interpolation True/False
    -   interp<str>: interpolation method; supported_interp = ['Trilinear','Nearest']
    
    Returns
    -------
    -   out: transformed input image batch. Tensor of size (B, H*, W*, D*, C).
    
    Use
    ---
    
    out = Apply3DTform(input_size, output_size)(Img, Tform, padding, interp)
    
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    def __init__(self, input_size, output_size, **kwargs):
        self.output_size = output_size
        self.input_size = input_size
        super(Apply3DTform, self).__init__(**kwargs)
        
    def get_config(self):
        return {
                'output_size' : self.output_size, 
        }
    
    def compute_output_shape(self, input_shape):
        self.height, self.width, self.depth = self.output_size
        channels = input_shape[-1]
        return (None, self.height, self.width, self.depth, channels)
        
    def call(self, Img, Tform, padding=False, interp='Trilinear'):
    
        supported_interp = ['Trilinear','Nearest']
        try:
            test = interp in supported_interp
            if test == False:
                raise ValueError('Wrong Keyword For Interp Method')
        except ValueError:
            print("Supported interp keywords : 'Trilinear' (Default) or 'Nearest'")
       
        output = self._Imwarp(Img, Tform, self.input_size, self.output_size, padding, interp)
        return output
    
    def _interpolate(self, Img, x_s, y_s, z_s, method):
        # Padding for last row/col/slice
        padding = [[0,0],
                   [0,1], 
                   [0,1], 
                   [0,1],
                   [0,0]]
        Img = tf.pad(Img, padding)
        # grab input dimension  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        D = tf.shape(Img)[3]
        # Def max indx
        max_x = tf.cast(H , 'int32')-1
        max_y = tf.cast(W , 'int32')-1
        max_z = tf.cast(D , 'int32')-1
        # resacle x_s, y_s, z_s to [0, W-2], [0, H-2], [0, D-2]
        x = tf.cast(x_s, 'float32')
        y = tf.cast(y_s, 'float32')
        z = tf.cast(z_s, 'float32')
        x = .5*(x + 1)*tf.cast(max_x-1, 'float32')
        y = .5*(y + 1)*tf.cast(max_y-1, 'float32')
        z = .5*(z + 1)*tf.cast(max_z-1, 'float32')
        # grab corners points for each (x_i, y_i, z_i)
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')        
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1
        #clip to range [0, W-1], [0, H-1], [0, D-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x) 
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y) 
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)
        # get pixel value at corner coords
        I000 = self._get_pixel_value(Img, x0, y0, z0)
        I001 = self._get_pixel_value(Img, x0, y0, z1)
        I010 = self._get_pixel_value(Img, x0, y1, z0)
        I100 = self._get_pixel_value(Img, x1, y0, z0)
        I011 = self._get_pixel_value(Img, x0, y1, z1)
        I101 = self._get_pixel_value(Img, x1, y0, z1)
        I110 = self._get_pixel_value(Img, x1, y1, z0)
        I111 = self._get_pixel_value(Img, x1, y1, z1)
        # recast as float for calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32') 
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        z0 = tf.cast(z0, 'float32')
        z1 = tf.cast(z1, 'float32')
        # trilinear interp
        if method == 'Trilinear':  
            W000 = (x1-x)*(y1-y)*(z1-z)
            W001 = (x1-x)*(y1-y)*(z-z0)
            W010 = (x1-x)*(y-y0)*(z1-z)
            W100 = (x-x0)*(y1-y)*(z1-z)
            W011 = (x1-x)*(y-y0)*(z-z0)
            W101 = (x-x0)*(y1-y)*(z-z0)
            W110 = (x-x0)*(y-y0)*(z1-z)
            W111 = (x-x0)*(y-y0)*(z-z0)
        # Nearest neighbour
        elif method == 'Nearest':
            W000 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))*np.float32((z-z0)<(z1-z))
            W001 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))*np.float32((z1-z)<(z-z0))
            W010 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))*np.float32((z-z0)<(z1-z))
            W100 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))*np.float32((z-z0)<(z1-z))
            W011 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))*np.float32((z1-z)<(z-z0))
            W101 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))*np.float32((z1-z)<(z-z0))
            W110 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))*np.float32((z-z0)<(z1-z))
            W111 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))*np.float32((z1-z)<(z-z0)) 
        #add dim
        W000 = tf.expand_dims(W000, axis = 4)
        W001 = tf.expand_dims(W001, axis = 4)
        W010 = tf.expand_dims(W010, axis = 4)
        W100 = tf.expand_dims(W100, axis = 4)
        W011 = tf.expand_dims(W011, axis = 4)
        W101 = tf.expand_dims(W101, axis = 4)
        W110 = tf.expand_dims(W110, axis = 4)
        W111 = tf.expand_dims(W111, axis = 4)
        # return interp Output 
        out = tf.add_n([W000*I000, W001*I001, W010*I010, W100*I100, W011*I011,  W101*I101, W110*I110, W111*I111])
        
        return out  
             
    def _meshgrid(self, height, width, depth, vector, matrix):
        # grab nb batchs  
        batchs = tf.shape(vector)[0]
        # def grid
        ax, ay, az = tf.linspace(-1., 1., height), tf.linspace(-1., 1., width), tf.linspace(-1., 1., depth)        
        x_t, y_t, z_t = tf.meshgrid(ax, ay, az)
                
        x_t = tf.keras.backend.permute_dimensions(x_t, (1,0,2))
        y_t = tf.keras.backend.permute_dimensions(y_t, (1,0,2))
        z_t = tf.keras.backend.permute_dimensions(z_t, (1,0,2))
        # flaten and reshape to [x_t, y_t, z_t] (non homogeneous form)
        x_t_flat, y_t_flat, z_t_falt = tf.reshape(x_t, [-1]), tf.reshape(y_t, [-1]), tf.reshape(z_t, [-1])
        sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_falt])  
        # repeat sampling grid batchs time    
        sampling_grid = tf.expand_dims(sampling_grid, axis = 0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([batchs, 1, 1]))
        # cast tform to float for linalg ops
        # Translation
        vector = tf.expand_dims(vector, axis = -1)
        vector = tf.tile(vector, tf.stack([1, 1, height*width*depth]))
        vector = tf.cast(vector, 'float32')    
        # Rot/Scale/Shear
        matrix = tf.cast(matrix, 'float32')
        # Grid
        sampling_grid = tf.cast(sampling_grid, 'float32')
        # batch multiply tform sample grid
        # batchs_grids = tf.linalg.matmul(tf.linalg.inv(matrix), tf.add(sampling_grid, vector)) # shape(B, 3, H*W*D)
        batchs_grids = tf.add(tf.linalg.matmul(matrix, sampling_grid), vector) # shape(B, 3, H*W*D)
        # reshape to (B, 3, H, W, D)
        batchs_grids = tf.reshape(batchs_grids[:, :, :], [batchs, 3, height, width, depth])
        # return T grid
        return batchs_grids
        
    def _get_pixel_value(self, Img, x, y, z):
        # grab grid dim  
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        D = tf.shape(x)[3]
        # enumerate Img in batch
        batch_indx = tf.range(0, B)
        # reshape batch indx
        batch_indx = tf.reshape(batch_indx, (B, 1, 1, 1))
        batch_indx = tf.tile(batch_indx, (1, H, W, D))
        # stack coordinates
        indices = tf.stack([batch_indx, x, y, z], 4)
        # return intensity at (x, y, z)
        return tf.gather_nd(Img, indices)  
    
    def _Imwarp(self, Img, Tform, input_size, output_size, padding, interp):
        # grab Img size  
        B = tf.shape(Img)[0]
        # generate Id 
        # Id = tf.tile(tf.expand_dims(tf.eye(3), axis=0), [B,1,1])
        # Grab and Reshape Translation
        V = tf.reshape(Tform[:, 9:], [B, 3]) 
        V = tf.cast(V, 'float32')
        # Grab and Reshape Rot/Scale/Shear
        M = tf.reshape(Tform[:, :9], [B, 3, 3])
        M = tf.cast(M, 'float32')   
        # generate grids
        indx_grid = self._meshgrid(height = output_size[0], width = output_size[1], depth = output_size[2], vector=V, matrix=M)
        x_s = indx_grid[:,0,:,:,:]
        y_s = indx_grid[:,1,:,:,:]
        z_s = indx_grid[:,2,:,:,:]
        # padding if requested
        if padding:
            H = input_size[0]
            W = input_size[1]
            D = input_size[2]
            
            top_pad = np.int32(np.ceil((output_size[0]-H)/2))
            bottom_pad = np.int32(np.floor((output_size[0]-H)/2))
            left_pad = np.int32(np.ceil((output_size[1]-W)/2))
            right_pad = np.int32(np.floor((output_size[1]-W)/2))
            first_pad = np.int32(np.ceil((output_size[2]-D)/2))
            last_pad = np.int32(np.floor((output_size[2]-D)/2))
            
            pad = ((top_pad, bottom_pad), (left_pad, right_pad), (first_pad, last_pad))      
            
            Img = tf.keras.layers.ZeroPadding3D(padding=pad)(Img)
        #sample input with grid to get output
        output_Img = self._interpolate(Img, x_s, y_s, z_s, method=interp)
        return output_Img
    
   
#%% Apply 3D Displacement field to 3D image 
class Apply3DDispField(tf.keras.layers.Layer):
    
    """
    Spatial Transformer layer implementation for 3D displacement field
    
    The layer is composed of 4 elements:
        
    -   _Imwarp: takes the input 3D image Batch <Img> of size (B, H, W, D, 1), 
        the displacement field <DispField> of size (B, H*, W*, D*, 3), the output size 
        <output_size> of a sample Img in the batch (H*, W*, D*, 1), the padding 
        method <padding> and, the interpolation method <interp> 
        and outputs the warped output Batch of size (B, H*, W*, D*, 1)
    
    -   _meshgrid: generates a grid of (x, y, z) coordinates, wth regards to the 
        transformation, that correspond to a set of points where the 
        input should be sampled to produce the transformed output.
      
    -   _interpolate: takes as input the original image, the grid (x, y, z)
        and, and produces the output transformed image using the intrepolation 
        method
      
    -   _get_pixel_value: takes an image as input and outputs pixels value at 
        (x, y, z) coordinates.
      
      
    Input
    -----
    -   output_size<tuple>: size of the output image (H*, W*, D*, 1)
    -   Img<tensor>: the input 3D image batch of size (B, H, W, D, 1)
    -   DispField<tensor>: displacement field to apply to each input sample (B, H*, W*, D*, 3)
        Initialize to null.
    -   padding<bool>: apply padding before interpolation True/False
    -   interp<str>: interpolation method; supported_interp = ['Trilinear','Nearest']
    
    Returns
    -------
    -   out: transformed input image batch. Tensor of size (B, H*, W*, 1).
    
    Use
    ---
    
    out = Apply3DDispField(input_size, output_size)(Img, DispField, padding, interp)
    
    """
    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(Apply3DDispField, self).__init__(**kwargs)
        
    def get_config(self):
        return {
                'output_size' : self.output_size, 
        }
    
    def compute_output_shape(self, input_shape):
        self.height, self.width, self.depth = self.output_size
        channels = input_shape[-1]
        return (None, self.height, self.width, self.depth, channels)
        
    def call(self, Img, DispField, padding=False, interp='Trilinear'):
    
        supported_interp = ['Trilinear','Nearest']
        try:
            test = interp in supported_interp
            if test == False:
                raise ValueError('Wrong Keyword For Interp Method')
        except ValueError:
            print("Supported interp keywords : 'Trilinear' (Default) or 'Nearest'")
       
        output = self._Imwarp(Img, DispField, self.input_size, self.output_size, padding, interp)
        return output
    
    def _interpolate(self, Img, x_s, y_s, z_s, method):
        # Padding for last row/col/slice
        padding = [[0,0],
                   [0,1], 
                   [0,1], 
                   [0,1],
                   [0,0]]
        Img = tf.pad(Img, padding)
        # grab input dimension  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        D = tf.shape(Img)[3]
        # Def max indx
        max_x = tf.cast(H , 'int32')-1
        max_y = tf.cast(W , 'int32')-1
        max_z = tf.cast(D , 'int32')-1
        # resacle x_s, y_s, z_s to [0, W-2], [0, H-2], [0, D-2]
        x = tf.cast(x_s, 'float32')
        y = tf.cast(y_s, 'float32')
        z = tf.cast(z_s, 'float32')
        x = .5*(x + 1)*tf.cast(max_x-1, 'float32')
        y = .5*(y + 1)*tf.cast(max_y-1, 'float32')
        z = .5*(z + 1)*tf.cast(max_z-1, 'float32')
        # grab corners points for each (x_i, y_i, z_i)
        x0 = tf.cast(tf.round(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.round(y), 'int32')        
        y1 = y0 + 1
        z0 = tf.cast(tf.round(z), 'int32')
        z1 = z0 + 1
        #clip to range [0, W-1], [0, H-1], [0, D-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0, max_x)
        x1 = tf.clip_by_value(x1, 0, max_x) 
        y0 = tf.clip_by_value(y0, 0, max_y)
        y1 = tf.clip_by_value(y1, 0, max_y) 
        z0 = tf.clip_by_value(z0, 0, max_z)
        z1 = tf.clip_by_value(z1, 0, max_z)
        # get pixel value at corner coords
        I000 = self._get_pixel_value(Img, x0, y0, z0)
        I001 = self._get_pixel_value(Img, x0, y0, z1)
        I010 = self._get_pixel_value(Img, x0, y1, z0)
        I100 = self._get_pixel_value(Img, x1, y0, z0)
        I011 = self._get_pixel_value(Img, x0, y1, z1)
        I101 = self._get_pixel_value(Img, x1, y0, z1)
        I110 = self._get_pixel_value(Img, x1, y1, z0)
        I111 = self._get_pixel_value(Img, x1, y1, z1)
        # recast as float for calculation
        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32') 
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        z0 = tf.cast(z0, 'float32')
        z1 = tf.cast(z1, 'float32')
        # trilinear interp
        if method == 'Trilinear':  
            W000 = (x1-x)*(y1-y)*(z1-z)
            W001 = (x1-x)*(y1-y)*(z-z0)
            W010 = (x1-x)*(y-y0)*(z1-z)
            W100 = (x-x0)*(y1-y)*(z1-z)
            W011 = (x1-x)*(y-y0)*(z-z0)
            W101 = (x-x0)*(y1-y)*(z-z0)
            W110 = (x-x0)*(y-y0)*(z1-z)
            W111 = (x-x0)*(y-y0)*(z-z0)
        # Nearest neighbour
        elif method == 'Nearest':
            W000 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))*np.float32((z-z0)<(z1-z))
            W001 = np.float32((x-x0)<(x1-x))*np.float32((y-y0)<(y1-y))*np.float32((z1-z)<(z-z0))
            W010 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))*np.float32((z-z0)<(z1-z))
            W100 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))*np.float32((z-z0)<(z1-z))
            W011 = np.float32((x-x0)<(x1-x))*np.float32((y1-y)<(y-y0))*np.float32((z1-z)<(z-z0))
            W101 = np.float32((x1-x)<(x-x0))*np.float32((y-y0)<(y1-y))*np.float32((z1-z)<(z-z0))
            W110 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))*np.float32((z-z0)<(z1-z))
            W111 = np.float32((x1-x)<(x-x0))*np.float32((y1-y)<(y-y0))*np.float32((z1-z)<(z-z0)) 
        #add dim
        W000 = tf.expand_dims(W000, axis = 4)
        W001 = tf.expand_dims(W001, axis = 4)
        W010 = tf.expand_dims(W010, axis = 4)
        W100 = tf.expand_dims(W100, axis = 4)
        W011 = tf.expand_dims(W011, axis = 4)
        W101 = tf.expand_dims(W101, axis = 4)
        W110 = tf.expand_dims(W110, axis = 4)
        W111 = tf.expand_dims(W111, axis = 4)
        # return interp Output 
        out = tf.add_n([W000*I000, W001*I001, W010*I010, W100*I100, W011*I011,  W101*I101, W110*I110, W111*I111])
        
        return out  
             
    def _meshgrid(self, height, width, depth, displacement):
        # grab nb batchs  
        batchs = tf.shape(displacement)[0]
        # def grid
        ax, ay, az = tf.linspace(-1., 1., height), tf.linspace(-1., 1., width), tf.linspace(-1., 1., depth)        
        x_t, y_t, z_t = tf.meshgrid(ax, ay, az)
                
        x_t = tf.keras.backend.permute_dimensions(x_t, (1,0,2))
        y_t = tf.keras.backend.permute_dimensions(y_t, (1,0,2))
        z_t = tf.keras.backend.permute_dimensions(z_t, (1,0,2))
        # flaten and reshape to [x_t, y_t, z_t] (non homogeneous form)
        x_t_flat, y_t_flat, z_t_falt = tf.reshape(x_t, [-1]), tf.reshape(y_t, [-1]), tf.reshape(z_t, [-1])
        sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_falt])  
        # repeat sampling grid batchs time    
        sampling_grid = tf.expand_dims(sampling_grid, axis = 0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([batchs, 1, 1]))
        # reshape RegField
        indx_grid = tf.keras.backend.permute_dimensions(displacement, pattern=(0,4,1,2,3))
        indx_grid = tf.cast(indx_grid, 'float32')        
        dx = indx_grid[:,0,:,:]
        dy = indx_grid[:,1,:,:]
        dz = indx_grid[:,2,:,:]
        dx_flat, dy_flat, dz_flat = tf.reshape(dx, [batchs, height*width*depth]), tf.reshape(dy, [batchs, height*width*depth]), tf.reshape(dz, [batchs, height*width*depth])
        d_grid = tf.stack([dx_flat, dy_flat, dz_flat], axis=1)          
        # Apply Displacement 
        batchs_grids = sampling_grid - d_grid
        #reshape to (B, 2, H, W, D)
        batchs_grids = tf.reshape(batchs_grids, [batchs, 3, height, width, depth])
        # return T grid
        return batchs_grids
        
    def _get_pixel_value(self, Img, x, y, z):
        # grab grid dim  
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        D = tf.shape(x)[3]
        # enumerate Img in batch
        batch_indx = tf.range(0, B)
        # reshape batch indx
        batch_indx = tf.reshape(batch_indx, (B, 1, 1, 1))
        batch_indx = tf.tile(batch_indx, (1, H, W, D))
        # stack coordinates
        indices = tf.stack([batch_indx, x, y, z], 4)
        # return intensity at (x, y, z)
        return tf.gather_nd(Img, indices)  
    
    def _Imwarp(self, Img, RegField, input_size, output_size, padding, interp):
        # grab Img size  
        H = tf.shape(Img)[1]
        W = tf.shape(Img)[2]
        D = tf.shape(Img)[3]
        # generate grids
        indx_grid = self._meshgrid(height = output_size[0], width = output_size[1], depth = output_size[2], displacement=RegField)
        x_s = indx_grid[:,0,:,:,:]
        y_s = indx_grid[:,1,:,:,:]
        z_s = indx_grid[:,2,:,:,:]
        # padding if requested
        if padding:
            H = input_size[0]
            W = input_size[1]
            D = input_size[2]
            
            top_pad = np.int32(np.ceil((output_size[0]-H)/2))
            bottom_pad = np.int32(np.floor((output_size[0]-H)/2))
            left_pad = np.int32(np.ceil((output_size[1]-W)/2))
            right_pad = np.int32(np.floor((output_size[1]-W)/2))
            first_pad = np.int32(np.ceil((output_size[2]-D)/2))
            last_pad = np.int32(np.floor((output_size[2]-D)/2))
            
            pad = ((top_pad, bottom_pad), (left_pad, right_pad), (first_pad, last_pad))      
            
            Img = tf.keras.layers.ZeroPadding3D(padding=pad)(Img)
        #sample input with grid to get output
        output_Img = self._interpolate(Img, x_s, y_s, z_s, method=interp)
        return output_Img
    

#%% Generate a batch of random affine tform matrices for 3D images
def _3Dtform_generation(batch=1, dim = (32, 32, 32), limit_t = [16,16,16],  limit_r=[20,20,20], limit_scale=0, limit_shear=0, distribution='uniform'):
    """
    _3Dtform_generation: generate a minibatch of random 3D tranformation matrices

    Input
    -----
    -   batch<int>: number of tforms to generate 
    -   dim<tuple>: Dimension of the volume to transform 
    -   limit_t<list of int>: Maximum of amplitude for the translation (in voxel)
    -   limit_r<list of float>: Maximum of amplitude for the rotation (in °)
    -   limit_scale<float>: Maximum of amplitude for the scaling
    -   distribution<str>: Distribution from which to draw tform params. 'uniform' or 'normal' are supported.
    
    Returns
    -------
    -   tform: tform batch. Tensor of size (B, 12).
    -   [tx, ty, tz]: list of translation for each tform.
    -   [rx, ry, rz]: list of rotation for each tform.
    -   [sx, sy, sz]: list of scale for each tform.
    
    Use
    ---
    
    tform, [tx, ty, tz], [rx, ry, rz], [sx, sy, sz] = _3Dtform_generation(batch = 10, 
                                                                          dim  (128,128,64), 
                                                                          limit_t = [32,32,16],  
                                                                          limit_r = [10, 10, 10], 
                                                                          limit_scale = 0, 
                                                                          limit_shear=0, 
                                                                          distribution='uniform')
    
    """
    supported_dist = ['uniform', 'normal']
    try:
        test = distribution in supported_dist
        if test == False:
            raise ValueError('Wrong Keyword argument For distribution ')
    except ValueError:
        print("Supported distribution : 'uniform','normal'")
          
    # tform Parameters Distribution 
    if distribution == 'uniform':
        # T             
        tx = tf.random.uniform([batch, 1], -limit_t[0], limit_t[0])/dim[0]*2
        ty = tf.random.uniform([batch, 1], -limit_t[1], limit_t[1])/dim[1]*2     
        tz = tf.random.uniform([batch, 1], -limit_t[2], limit_t[2])/dim[2]*2  
        # R                   
        rx = tf.random.uniform([batch, 1], -limit_r[0], limit_r[0])*(np.pi/180) #deg to rad
        ry = tf.random.uniform([batch, 1], -limit_r[1], limit_r[1])*(np.pi/180) #deg to rad
        rz = tf.random.uniform([batch, 1], -limit_r[2], limit_r[2])*(np.pi/180) #deg to rad 
        # Scale
        scx = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
        scy = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
        scz = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
        # Shear
        shx = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
        shy = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
        shz = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
    elif distribution == 'normal':
        # T             
        tx = tf.random.normal([batch, 1], 0, (limit_t[0]/dim[0]*2)/3)
        ty = tf.random.normal([batch, 1], 0, (limit_t[1]/dim[1]*2)/3)    
        tz = tf.random.normal([batch, 1], 0, (limit_t[2]/dim[2]*2)/3)  
        # R                   
        rx = tf.random.normal([batch, 1], 0, (limit_r[0]*(np.pi/180))/3) #deg to rad
        ry = tf.random.normal([batch, 1], 0, (limit_r[1]*(np.pi/180))/3)  #deg to rad
        rz = tf.random.normal([batch, 1], 0, (limit_r[2]*(np.pi/180))/3)  #deg to rad 
        # Scale
        scx = tf.random.normal([batch, 1], 0, limit_scale/3)
        scy = tf.random.normal([batch, 1], 0, limit_scale/3)
        scz = tf.random.normal([batch, 1], 0, limit_scale/3)
        # Shear
        shx = tf.random.normal([batch, 1], 0, limit_shear/3)
        shy = tf.random.normal([batch, 1], 0, limit_shear/3)
        shz = tf.random.normal([batch, 1], 0, limit_shear/3)
    
    # Translation  
    trans = tf.concat([tx, ty, tz], axis=-1)
    
    # Rotation    
    cosx = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(rx)
    sinx = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,-1,0,1,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(rx)
    rotx = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])+cosx+sinx
                   
    cosy = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(ry)
    siny = tf.tile(tf.cast(tf.expand_dims([0,0,1,0,0,0,-1,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(ry)
    roty = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])+cosy+siny
    
    cosz = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(rz)
    sinz = tf.tile(tf.cast(tf.expand_dims([0,-1,0,1,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(rz)
    rotz = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])+cosz+sinz
    
    # Combine rotation
    rot = tf.linalg.matmul(tf.reshape(rotz, [batch, 3, 3]), tf.linalg.matmul(tf.reshape(roty, [batch, 3, 3]), tf.reshape(rotx, [batch, 3, 3])))
   
    # Scaling
    scalex = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*scx
    scaley = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*scy
    scalez = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*scz
    
    scale = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32'), [batch, 1])+scalex+scaley+scalez
    
    # Shear
    shearx = tf.tile(tf.cast(tf.expand_dims([0,shy,shz,0,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])
    sheary = tf.tile(tf.cast(tf.expand_dims([0,0,0,shx,0,shz,0,0,0], axis=0), dtype='float32'), [batch, 1])
    shearz = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,0,shx,shy,0], axis=0), dtype='float32'), [batch, 1])
    
    shear = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32'), [batch, 1])+shearx+sheary+shearz
    
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(shear, [batch, 3, 3]), tf.linalg.matmul(tf.reshape(scale, [batch, 3, 3]), rot))
    
    tform = tf.concat([tf.reshape(matrix, [batch, 9]), trans], axis=-1)
    
    return tform, [tx, ty, tz], [rx, ry, rz], [scx, scy, scz], [shx, shy, shz]

#%% Generate a single 3D tform matrix from input params
def Make_3Dtform(dim = [32,32,32], trans = [16,16,16], rot = [20,20,20], scale = [0, 0, 0], shear=[0, 0, 0]):
    """
    Make_3Dtform: generates a 3D tranformation matrix with specific params
    
    Input
    -----
    -   dim<tuple>: Dimension of the volume to transform 
    -   trans<list of int>: translation (in voxel)
    -   rot<list of float>: rotation (in °)
    -   scale<list of float>: scaling
    -   shear<list of float>: shear

    
    Returns
    -------
    -   tform: Tensor of size (1, 6).
    
    Use
    ---
    
    tform = Make_3Dtform(dim = (128,128,64), 
                         trans = [32,32,16],  
                         rot = [10, 10, 10], 
                         scale = [0, 0, 0], 
                         shear = [0, 0, 0])
    
    """
    # Translation              
    trans = tf.cast(trans, 'float32')/tf.cast(dim, 'float32')*2
        
    # Rotation                   
    rx = rot[0]*(np.pi/180) #deg to rad
    ry = rot[1]*(np.pi/180) #deg to rad
    rz = rot[2]*(np.pi/180) #deg to rad 

    cosx = tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,1], axis=0), dtype='float32')*tf.math.cos(rx)
    sinx = tf.cast(tf.expand_dims([0,0,0,0,0,-1,0,1,0], axis=0), dtype='float32')*tf.math.sin(rx)
    rotx = tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32')+cosx+sinx
                   
    cosy = tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,1], axis=0), dtype='float32')*tf.math.cos(ry)
    siny = tf.cast(tf.expand_dims([0,0,1,0,0,0,-1,0,0], axis=0), dtype='float32')*tf.math.sin(ry)
    roty = tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32')+cosy+siny
    
    cosz = tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,0], axis=0), dtype='float32')*tf.math.cos(rz)
    sinz = tf.cast(tf.expand_dims([0,-1,0,1,0,0,0,0,0], axis=0), dtype='float32')*tf.math.sin(rz)
    rotz = tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32')+cosz+sinz
    
    # Combine rotation
    rot = tf.linalg.matmul(tf.reshape(rotz, [3, 3]), tf.linalg.matmul(tf.reshape(roty, [3, 3]), tf.reshape(rotx, [3, 3])))
   
    # Scaling
    scx = scale[0]
    scy = scale[1]
    scz = scale[2]
    
    scalex = tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32')*scx
    scaley = tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32')*scy
    scalez = tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32')*scz
    
    scale = tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32')+scalex+scaley+scalez
           
    # Shear
    shx = shear[0]
    shy = shear[1]
    shz = shear[2]
    
    shearx = tf.cast(tf.expand_dims([0,shy,shz,0,0,0,0,0,0], axis=0), dtype='float32')
    sheary = tf.cast(tf.expand_dims([0,0,0,shx,0,shz,0,0,0], axis=0), dtype='float32')
    shearz = tf.cast(tf.expand_dims([0,0,0,0,0,0,shx,shy,0], axis=0), dtype='float32')
    
    shear = tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32')+shearx+sheary+shearz
    
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(shear, [3, 3]), tf.linalg.matmul(tf.reshape(scale, [3, 3]), rot))
    
    tform = tf.concat([tf.reshape(matrix, [1, 9]), tf.expand_dims(trans, axis=0)], axis=-1)
    
    return tform     
    
    
    
    
    