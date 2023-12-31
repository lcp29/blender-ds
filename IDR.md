## IDR Dataset Specification

#### Directory Structure
```
dataset_root---image---000.png
             |       |-001.png
             |       |-002.png
             |       ...
             |
             |--mask---000.png
             |       |-001.png
             |       ...
             \--camera_xxx.npz
```

#### Camera Convention

![camera-coord](static/camera-coord.png)

The camera points to +z axis and the right hand side is +x axis. The camera space is in right-hand coordinates.

To transform a world-space object to the camera space, we left-multiply its homogeneous location $[x,y,z,1]^\mathrm T$ with a $3\times4$ matrix $[R|t]$, which represents the orthogonal rotation and the translation in the transform. To obtain $[R|t]$, we derive the inverse of the matrix $\left[\begin{matrix}\mathbf x&\mathbf y&\mathbf z&\mathbf t\\0&0&0&1\end{matrix}\right]$ in which $\mathbf x$, $\mathbf y$, $\mathbf z$ is the camera orientations (in the above convention) and $\mathbf t$ is the location of the camera's central point.

After above transform we have the object in camera space $[x',y',z']^\mathrm T$, but we are representing it in the form of canonical image projection space:
$$
\left[x',y',z'\right]^\mathrm T=z'\left[x_p,y_p,1\right]^\mathrm T,
$$
where $x_p=x'/z'$, $y_p=y'/z'$. This is a $45^\circ$ perspective projection space with $f_x=f_y=1$. Note that we don't need near and far plane because we are flatting the whole space into $z_p=1$ plane thus no need to order them.

To further transform it into image window $(u,v)\in[0, \mathrm{width}]\times[0, \mathrm{height}]$, we have
$$
\begin{align}
&u=f_xx_p+c_x,\\
&v=f_yy_p+c_y,\\
&z_{img}=1
\end{align}
$$
to map the points to the image window points (different from ordinary perspective projection matrices we make $z=1$ to "compress" the whole scene to the $z=1$ plane), where $f_x$ and $f_y$ are x and y focal lengths (can be taken as the distance from th film to the focal point when the file's of size $(H, W)$) and $c_x$, $c_y$ are the UV coordinates of the image central point. There is a corresponding matrix with the above transform:
$$
[u,v,1]^\mathrm T=K[x_p,y_p,1]^\mathrm T, \text{where}\ K=\left[\begin{matrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{matrix}\right].
$$
The UV coordinates start from the top-left corner of the image and reaches their maximums at the bottom-right corner as $(\text{width}-1,\text{height}-1)$. Assume we have a fovX $\alpha$ and image size $(H, W)$, at $z'=1$ we have $x'=\tan(\alpha/2)$ and we want to enlarge it to $W/2$ so we have the focal length $\frac{W}{2\tan(\alpha/2)}$ and $c_x=(W-1)/2$. 

#### NPZ File Format

An .npz file contains several "files" that can be indexed by their names. A .npz file from the dataset has the following files:

```
world_mat_0, scale_mat_0, world_mat_1, scale_mat_1, ...
```

each of which corresponds with the two image files in the image and mask folders.

##### world_mat

A world_mat is a numpy array of size [4, 4] as a VP matrix described above:
$$
\text{world\_mat}=
\left[
\begin{matrix}
f_x & 0 & c_x & 0 \\
0 & f_y & c_y & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{matrix}
\right]
\cdot
\left[
\begin{matrix}
\mathbf x & \mathbf y & \mathbf z & \mathbf t \\
0 & 0 & 0 & 1
\end{matrix}
\right]^{-1},\\
\text{where\ }
f_x=\frac{W}{2\cdot\tan(\text{fovX/2})},
c_x=\frac{W-1}{2},\cdots
$$

##### scale_mat

In NeuS we assume the whole scene is in a unit bounding sphere located in the center. If the actual bounding sphere of the scene is of radius $R$ and centered at $(x_r,y_r,z_r)$, we need to do an inverse transform to move it back to the unit sphere, or we can just write it down as transforming the scene from the unit sphere at the center to the actual bounding sphere:
$$
\text{scale\_mat}=
\left[
\begin{matrix}
R & 0 & 0 & x_r \\
0 & R & 0 & y_r \\
0 & 0 & R & z_r \\
0 & 0 & 0 & 1
\end{matrix}
\right]
$$
