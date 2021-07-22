#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:36:17 2021

@author: karthik
"""

import tensorflow_addons as tfa
import tensorflow as tf

def Mlp(in_features, hidden_features=None, out_features=None, activation=None, dropout=0.0, **kwargs):
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    input_shape = (None, in_features, )
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Dense(units=hidden_features, activation=activation)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(units=out_features, activation=None)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, x, name='mlp', **kwargs)

def PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs):
    num_patches = (img_size // patch_size) ** 2
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)
    return tf.keras.Model(inputs, x, name='patch_embed', **kwargs)

def Attention(dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, **kwargs):
    head_dim = dim // num_heads
    scale = qk_scale or head_dim ** -0.5
    inputs = tf.keras.layers.Input((197, dim))
    x = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias)(inputs)
    x = tf.keras.layers.Reshape((197,3,num_heads, dim//num_heads))(x)
    x = tf.transpose(x, perm=[2, 0, 3, 1, 4])
       
    q, k, v = tf.split(x, 3, 0)
    q = q[0]
    k = k[0]
    v = v[0]
    #print(q.shape)
    k = tf.keras.layers.Permute((1, 3, 2))(k)
    #print(k.shape)
    attn = tf.keras.layers.Lambda(lambda x : tf.linalg.matmul(x[0], x[1]) * scale)([q, k])
    attn = tf.keras.layers.Softmax()(attn)
    attn = tf.keras.layers.Dropout(attn_drop)(attn)
    v = tf.keras.layers.Lambda(lambda x : tf.linalg.matmul(x[0], x[1]))([attn, v])
    x = tf.keras.layers.Permute((2, 1, 3))(v)
    x = tf.keras.layers.Reshape((197, dim))(x)
    x = tf.keras.layers.Dense(dim)(x)
    x = tf.keras.layers.Dropout(proj_drop)(x)
    return tf.keras.Model(inputs, outputs=[x, attn], name='attention', **kwargs)

def Block(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=tf.nn.gelu, **kwargs):
    mlp_hidden_dim = int(dim * mlp_ratio)
    inputs = tf.keras.layers.Input((197, dim))
    attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                     qk_scale=qk_scale, 
                     attn_drop=attn_drop, proj_drop=drop)
    x = tf.keras.layers.LayerNormalization()(inputs)
    x , attention = attn(x)
    x = tf.keras.layers.Add()([x, inputs])
    y = tf.keras.layers.LayerNormalization()(x)

    y = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            activation=act_layer, dropout=drop)(y)

    

    x = tf.keras.layers.Add()([x, y])
    return tf.keras.Model(inputs, [x, attention], **kwargs)

def Vit(img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., **kwargs):
    inputs = tf.keras.layers.Input((img_size, img_size, in_chans))
    x = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)(inputs)
    cls_token = tf.keras.layers.Lambda(lambda x : tf.random.truncated_normal((tf.shape(x)[0], 1, x.shape[2]), stddev=0.2), **{'name':'class_token'})(x)
    x = tf.keras.layers.Concatenate(axis=1)([cls_token, x])

    pos_embed = tf.keras.layers.Lambda(lambda x: tf.tile(tf.random.truncated_normal((1, x.shape[1], x.shape[2]), stddev=0.2), (tf.shape(x)[0], 1, 1)),
                      **{'name':'position_encoding'})(x)
    
    x = tf.keras.layers.Add()([x, pos_embed])
    x = tf.keras.layers.Dropout(drop_rate)(x)
    attentions = []
    for i in range(depth):
        x, attn = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, **{'name':'block_{}'.format(i+1)})(x)
        attentions.append(attn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Lambda(lambda x:x[:, 0])(x)
    return tf.keras.Model(inputs, [x, attentions], name="vision_transformer", **kwargs)

def get_last_selfattention(vit_model, x):
    result, attentions = vit_model(x)
    return attentions[-1]

def vit_tiny(patch_size=16):
    model = Vit(img_size=224, patch_size=patch_size, in_chans=3, num_classes=0, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., )
    return model

def vit_small(patch_size=16):
    model = Vit(img_size=224, patch_size=patch_size, in_chans=3, num_classes=0, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., )
    return model

def vit_base(patch_size=16):
    model = Vit(img_size=224, patch_size=patch_size, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., )
    return model

def DINOHead(in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3,
            hidden_dim=2048, bottleneck_dim=256, **kwargs):
    nlayers = max(nlayers, 1)
    inputs = tf.keras.layers.Input((in_dim, ))
    if nlayers==1:
        x = tf.keras.layers.Dense(bottleneck_dim)(inputs)
    else:
        x = tf.keras.layers.Dense(hidden_dim)(inputs)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
        x = tf.keras.layers.Lambda(lambda x : tf.keras.activations.gelu(x))(x)
        for _ in range(nlayers - 2):
            x = tf.keras.layers.Dense(hidden_dim)(x)
            if use_bn:
                x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
            x = tf.keras.layers.Lambda(lambda x : tf.keras.activations.gelu(x))(x)
        x = tf.keras.layers.Dense(bottleneck_dim)(x)
    x = tf.keras.layers.Lambda(lambda x:tf.math.l2_normalize(x, axis=0), **{'name':'L2_Norm'})(x)
    dense = tf.keras.layers.Dense(out_dim, use_bias=False)
    last_layer = tfa.layers.WeightNormalization(dense)
    last_layer.build(x.shape)
    
    if norm_last_layer:
        last_layer.g = last_layer.add_weight(
                name="g",
                shape=(last_layer.layer_depth,),
                initializer="ones",
                dtype=last_layer.layer.kernel.dtype,
                trainable=False,
            )

    x = last_layer(x)

    return tf.keras.Model(inputs, x, name='dino_head', **kwargs)
    