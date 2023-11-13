import numpy as np
from PIL import Image

PIXEL_SCALE = 3
IMG_SCALE = 24

BASE_COLOUR = (0,0,255)
CHEQUER_V = 230

FOOD_BASE = (  0,255,255)
FOOD_RING = (  0,128,255)
FOOD_LVLS = (  0,255,204)

AGENT_BASE = (140,255,255)
AGENT_RING = (140,128,255)
AGENT_LVLS = (140,255,204)

ring_points = (
    [(r, PIXEL_SCALE-1) for r in range(PIXEL_SCALE-1)] +
    [(PIXEL_SCALE-1, c) for c in reversed(range(1,PIXEL_SCALE))] +
    [(r, 0          ) for r in reversed(range(1,PIXEL_SCALE))] +
    [(0, c          ) for c in range(PIXEL_SCALE-1)]
)

def _pixel_to_slice(pixel):
    return slice(pixel*PIXEL_SCALE, (pixel+1)*PIXEL_SCALE)


def _color_to_arr(color):
    return np.expand_dims(np.array(color, dtype=np.uint8), (1,2))


def _food_pixel(lvl):
    """Builds a food sprite of given level `lvl`."""
    pixel = np.tile(
        _color_to_arr(FOOD_BASE),
        (PIXEL_SCALE, PIXEL_SCALE)
        )
    #draw the level indicator ring:
    pixel[:, 0,:] = _color_to_arr(FOOD_RING).reshape(3,1)
    pixel[:,-1,:] = _color_to_arr(FOOD_RING).reshape(3,1)
    pixel[:,:, 0] = _color_to_arr(FOOD_RING).reshape(3,1)
    pixel[:,:,-1] = _color_to_arr(FOOD_RING).reshape(3,1)
    ring_start = PIXEL_SCALE//2
    for l in range(lvl):
        point = ring_points[(l+ring_start)%len(ring_points)]
        pixel[(slice(None), point[1], point[0])] = _color_to_arr(FOOD_LVLS).squeeze()
    return pixel


def _agent_pixel(lvl):
    """Builds a agent sprite of given level `lvl`."""
    pixel = np.tile(
        _color_to_arr(AGENT_BASE),
        (PIXEL_SCALE, PIXEL_SCALE)
        )
    #draw the level indictator ring:
    pixel[:, 0,:] = _color_to_arr(AGENT_RING).reshape(3,1)
    pixel[:,-1,:] = _color_to_arr(AGENT_RING).reshape(3,1)
    pixel[:,:, 0] = _color_to_arr(AGENT_RING).reshape(3,1)
    pixel[:,:,-1] = _color_to_arr(AGENT_RING).reshape(3,1)
    ring_start = PIXEL_SCALE//2
    for l in range(lvl):
        point = ring_points[(l+ring_start)%len(ring_points)]
        pixel[(slice(None), point[1], point[0])] = _color_to_arr(AGENT_LVLS).squeeze()
    return pixel


def render(env):
    """Renders the envrionment."""
    base_pixel = np.tile(
        _color_to_arr(BASE_COLOUR),
        (PIXEL_SCALE, PIXEL_SCALE)
    )
    field_size = env.field_size
    img = np.tile(base_pixel, field_size)
    # chequer
    for y in range(field_size[0]):
        for x in range(field_size[1]):
            if (x-y)%2 == 0:
                r = _pixel_to_slice(y)
                c = _pixel_to_slice(x)
                img[2,r,c] = CHEQUER_V

    # Food
    for (y,x) in zip(*np.nonzero(env.field)):
        r = _pixel_to_slice(y)
        c = _pixel_to_slice(x)
        l = env.field[y,x]
        img[:,r,c] = _food_pixel(l)

    # Agents
    for agent, pos in env.pos.items():
        r = _pixel_to_slice(pos[0])
        c = _pixel_to_slice(pos[1])
        l = env.agent_levels[agent]
        img[:,r,c] = _agent_pixel(l)

    rgb_image = Image.fromarray(
            img.transpose((1,2,0)), mode="HSV"
        ).convert(
            "RGB"
        ).resize(
            (IMG_SCALE*img.shape[2], IMG_SCALE*img.shape[1]),
            resample=Image.Resampling.NEAREST,
        )
    return np.asarray(rgb_image)
