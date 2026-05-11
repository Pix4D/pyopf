#!/usr/bin/env python

import numpy as np

from pyopf.io import load
from pyopf.resolve import resolve

camera_matches_count = np.zeros((3, 3), dtype=int)
camera_track_size = np.zeros(3, dtype=int)
def print_tracks(tracks):
    
    # all tracks camera ids
    #for idx in tracks.matches.camera_ids:
    #    print("camera id: ", idx, tracks.matches.camera_ids[idx])
    global camera_matches_count
    global camera_track_size
    for idx, p3 in enumerate(tracks.position):
        idx_range = tracks.matches.point_index_ranges[idx] # size 2 array of the track with [0] camera index and [1] size of track 
        uid_indices = tracks.matches.camera_ids[
            idx_range[0] : idx_range[0] + idx_range[1]
        ].flatten()
        
        camera_uids = [tracks.matches.camera_uids[i] for i in uid_indices]
        image_points = tracks.matches.image_points.pixelCoordinates[
            idx_range[0] : idx_range[0] + idx_range[1]
        ]
        
        camera_track_size[idx_range[1]] +=1
        for i in range(len(uid_indices)-1):
            ci1 = uid_indices[i]
            ci2 = uid_indices[i+1]
            camera_matches_count[ci2,ci1] +=1
            camera_matches_count[ci1,ci2] +=1
            
        # point_texts = []
        # for point_idx in range(len(camera_uids)):
        #     uid = camera_uids[point_idx]
        #     p = image_points[point_idx]
        #     point_texts.append(f"UID {uid.int} {uid_indices[point_idx]} p2 {p}")
        # print(f"p3 {p3} reprojections {{" + ", ".join(point_texts) + "}")
        

def main():
    print("start")
    output_dir = 'C:/images/out/'
    input = 'C:/images/gatewing_opf/project.opf'
    print(output_dir, input)
    
    project = load(input)
    project = resolve(project)
    
    calibrated_camera_number = len(project.calibration.calibrated_cameras.cameras)
    global camera_matches_count
    global camera_track_size
    camera_matches_count = np.zeros((calibrated_camera_number,calibrated_camera_number))
    camera_track_size = np.zeros(calibrated_camera_number)
    print("calibrated cameras: ", calibrated_camera_number)
    for idx, calibration in enumerate(project.calibration_objs):
        print(f"Calibration {idx}")
        print_tracks(calibration.tracks.nodes[0])

    total_tracks=0
    for i in range(calibrated_camera_number):
        if camera_track_size[i] != 0:
            print("number of tracks with size:", i, "is", f"{camera_track_size[i]:.0f}")
            total_tracks += camera_track_size[i]
    print("total tracks: ", f"{total_tracks:.0f}")
    
    print("Matches between cameras:")
    for i in range(calibrated_camera_number):
        point_texts = []
        for j in range(calibrated_camera_number):
            point_texts.append(f"{camera_matches_count[i,j]:4.0f}")
        print(point_texts)       
        
    
    
    
if __name__ == "__main__":
    main()