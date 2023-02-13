import os.path


def read_result(alg_name):
    file_name = os.getcwd()
    result_path = os.path.join(file_name, 'runtime', alg_name, 'populations', 'results.txt').replace('\\', '/')
    best_error = 100.0
    best_loss = 100.0
    with open(result_path, 'r') as f:
        line = f.readline().strip()
        while line:
            error = line.split('=')[1]
            loss = line.split('=')[2]
            if float(error) < best_error:
                best_error = float(error)
            if float(loss) < best_loss:
                best_loss = float(loss)
            line = f.readline().strip()
        f.close()
    return best_error, best_loss


def write_result(alg_name, best_error, best_loss):
    file_name = os.getcwd()
    result_path = os.path.join(file_name, 'runtime', 'results.txt').replace('\\', '/')
    if not os.path.exists(result_path):
        file = open(result_path, 'w')
        file.close()
    fd = open(result_path, "a")
    res = f'\n{alg_name}={best_error},{best_loss}'
    fd.write(res)
    fd.close()

