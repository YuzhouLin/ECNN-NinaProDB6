def train():
    global x
    if 0.5 < x:
        x = 0.5
    print(x)
    return

if __name__ == "__main__":
    global X
    x = 100
    train()
    print(x)