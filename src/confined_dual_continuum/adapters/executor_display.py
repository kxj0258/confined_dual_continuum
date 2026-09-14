"""主臂显示和离线视频；标记仅添加到渲染场景，不参与动力学。"""
import numpy as np


def add_tracking_geometry(mj, scene, target, actual, reference_path, actual_path):
    def line(start, end, color, width=2.):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(end-start) < 1e-12:
            return
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                       np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_LINE, width, start, end)
        scene.ngeom += 1
    def ball(point, radius, color):
        if scene.ngeom >= scene.maxgeom:
            return
        mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, radius), point, np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        scene.ngeom += 1
    for points, color in ((reference_path, [.6, .6, .6, 1.]), (actual_path, [.1, .8, .25, 1.])):
        points = np.asarray(points)
        if len(points) > 300:
            points = points[np.linspace(0, len(points)-1, 300).astype(int)]
        for a, b in zip(points[:-1], points[1:]):
            line(a, b, color)
    line(target, actual, [1., .65, .1, 1.], 3.)
    ball(actual, .0015, [.1, .9, .25, 1.])
    ball(target, .002, [1., .15, .1, 1.])


def configure_camera(camera, base_position):
    camera.lookat[:] = base_position+np.array([0., 0., .09])
    camera.distance, camera.azimuth, camera.elevation = .32, 135., -20.


def export_video(scene_path, frames, output_path, fps, width, height, reference_at,
                 reference_path, progress=None):
    """用独立 MjData 重建记录姿态，不推进物理，不改变实时 viewer 的最终状态。"""
    import mujoco
    import imageio.v2 as imageio
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'executor_tip')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'mobile_base_frame')
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    configure_camera(camera, data.site_xpos[base_id])
    path = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for i, frame in enumerate(frames):
                data.qpos[:] = frame.qpos
                data.qvel[:] = frame.qvel
                data.time = frame.time_s
                mujoco.mj_forward(model, data)
                actual = data.site_xpos[tip_id].copy()
                path.append(actual)
                renderer.update_scene(data, camera=camera)
                add_tracking_geometry(mujoco, renderer.scene, reference_at(frame.time_s)[0], actual,
                                      reference_path, path)
                writer.append_data(renderer.render())
                if progress is not None and (i % 20 == 0 or i+1 == len(frames)):
                    progress(f'生成视频 {i+1}/{len(frames)} 帧')
