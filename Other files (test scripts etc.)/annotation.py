import json

def get_annotation(ann_path):
    with open(ann_path, 'r') as file:
        return json.load(file)
    
def get_bbs(annotation):
    """ Return data structure of bounding boxes ready for plot. """

    bbs = []

    bb_structure = annotation['Image_data']['boundingboxes']
    
    # there's a bug in the annotation structure, where if there is only
    # a single bb it's not written out as a list
    if not isinstance(bb_structure, list):
        print 'bad bb detected, %s' % annotation['Image_data']['Filename']
        bb_structure = [bb_structure]
    
    for bb in bb_structure:

        width = bb['corner_bottom_right_x'] - bb['corner_top_left_x']
        height = bb['corner_bottom_right_y'] - bb['corner_top_left_y']

        # get bottom left co-ordinate
        xy = (bb['corner_top_left_x'], bb['corner_top_left_y'])
        bbs.append((xy, width, height))

    return bbs
        
