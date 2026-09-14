"""仅保留双臂与基座的 MuJoCo 场景构建；交互与采集共用。"""
from pathlib import Path
import xml.etree.ElementTree as ET


def robot_only_scene(source: Path, destination: Path) -> None:
    """复用当前机器人资产，明确移除任务台和地面，不复制一份机器人参数。"""
    source = source.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find('worldbody')
    if world is None or world.find("body[@name='mobile_base']") is None:
        raise ValueError('源场景缺少 mobile_base')
    for child in list(world):
        if child.tag != 'light' and not (child.tag == 'body' and child.get('name') == 'mobile_base'):
            world.remove(child)
    compiler = root.find('compiler')
    for element in root.findall('./asset/*'):
        if 'file' in element.attrib:
            directory = compiler.get(element.tag + 'dir', '') if compiler is not None else ''
            element.set('file', str((source.parent / directory / element.get('file')).resolve()))
    if compiler is not None:
        for name in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(name, None)
    tree.write(destination, encoding='utf-8', xml_declaration=True)


