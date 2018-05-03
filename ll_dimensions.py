def main():
    SCALE  = 30.0
    VIEWPORT_W = 600
    VIEWPORT_H = 400
    W = VIEWPORT_W/SCALE
    H = VIEWPORT_H/SCALE
    CHUNKS = 11
    LEG_DOWN = 18
    chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
    helipad_x1 = chunk_x[CHUNKS//2-1]
    helipad_x2 = chunk_x[CHUNKS//2+1]
    helipad_y  = H/4
    print helipad_y

    def x_coord(pos_x):
        x_coord = (pos_x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
        return x_coord

    def y_coord(pos_y):
        y_coord = (pos_y - (helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2)
        return y_coord

    print VIEWPORT_W/SCALE/2

if __name__ == '__main__':
    main()
