from tests.city import myCity

if __name__ == '__main__':
    action = []
    # driver_grid_id, driver_id, order_grid_id, order_id, order = action
    for i in range(50):
        s_, reward, info = myCity.step(action)
        print(s_, reward, info)
