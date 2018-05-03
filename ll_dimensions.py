def main():
    SCALE  = 30.0
    VIEWPORT_W = 600
    VIEWPORT_H = 400
    W = VIEWPORT_W/SCALE
    H = VIEWPORT_H/SCALE
    CHUNKS = 11
    chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
    helipad_x1 = chunk_x[CHUNKS//2-1]
    helipad_x2 = chunk_x[CHUNKS//2+1]
    helipad_y  = H/4
    print helipad_y

if __name__ == '__main__':
    main()
