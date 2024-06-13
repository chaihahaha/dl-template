import inspect
from PIL import Image
import matplotlib.pyplot as plt
import dill
from functools import wraps
import numpy as np

def packplot(plotf):
    @wraps(plotf)
    def decorated_plotf(*args, **kwargs):
        import dill
        import inspect
        from PIL import Image
        import os
        import uuid
        n_pos_args = len(args)
        all_arg_names = list(inspect.signature(plotf).parameters.keys())
        pos_arg_names = all_arg_names[:n_pos_args]
        for i in range(n_pos_args):
            kwargs[pos_arg_names[i]] = args[i]
        assert 'filename' in kwargs
        filename = kwargs['filename']

        ret = plotf(**kwargs)

        plot_function_name = plotf.__name__
        try:
            plot_function_code = inspect.getsource(plotf)
        except OSError:
            print(f"Cannot find the code of function {plot_function_name}")
            return ret
        plot_function_code = clean_code(plot_function_code)

        newname = uuid.uuid4().hex + filename
        os.rename(filename, newname)
        img = Image.open(newname)
        packed_info = (kwargs, plot_function_code, plot_function_name)
        packed_bytes = dill.dumps(packed_info)
        img.save(filename, exif=packed_bytes)
        img.close()
        return ret

    return decorated_plotf

#def plot_random(x, y, filename, **kwargs):
#    plot_args = locals()
#    import dill
#    import inspect
#    from PIL import Image
#
#    import matplotlib.pyplot as plt
#    plt.plot(x, y)
#    plt.show()
#    plt.savefig(filename)
#
#    try:
#        plot_function_code = inspect.getsource(plot_random)
#        plot_function_name = 'plot_random'
#        img = Image.open(filename)
#        packed_info = (plot_args, plot_function_code, plot_function_name)
#        packed_bytes = dill.dumps(packed_info)
#        img.save(filename, exif=packed_bytes)
#    except OSError:
#        return
#    return

def clean_code(code_str):
    # Split the code into lines
    lines = code_str.splitlines()
    common_indent = ''
    min_len = min(len(l) for l in lines)
    should_break = False
    for i in range(min_len):
        cs = set()
        for j in range(len(lines)):
            c = lines[j][i]
            if c in ['\t',' ']:
                cs.add(c)
            else:
                should_break = True
        if should_break:
            break
        if len(cs)==1:
            common_indent += list(cs)[0]
    
    # Strip leading and trailing whitespace from each line
    cleaned_lines = [line[len(common_indent):] for line in lines]
    
    # Join the cleaned lines back into a single string with newlines
    cleaned_code_str = '\n'.join(cleaned_lines)
    
    return cleaned_code_str

def retrieve_plot(filename):
    img = Image.open(filename)
    img.load()
    packed_info = dill.loads(img.info['exif'][6:])
    plot_args, plot_function_code, plot_function_name = packed_info
    #assert plot_function_code.startswith("@packplot")
    plot_function_code = plot_function_code.split("\n", 1)[1]
    filename = plot_args['filename']
    plot_args['filename'] = f'tmp-{str(hash(filename))}' + filename
    exec(plot_function_code)
    exec(plot_function_name + '(**plot_args)')

def retrieve_data(filename):
    img = Image.open(filename)
    img.load()
    packed_info = dill.loads(img.info['exif'][6:])
    plot_args, plot_function_code, plot_function_name = packed_info
    return plot_args

def retrieve_code(filename):
    img = Image.open(filename)
    img.load()
    packed_info = dill.loads(img.info['exif'][6:])
    plot_args, plot_function_code, plot_function_name = packed_info
    return plot_function_code

if __name__ == '__main__':
    @packplot
    def plot_test_dec(x, y, filename):
        plt.plot(x, y)
        plt.show()
        plt.savefig(filename)
        return 0

    x = np.random.rand(10)
    y = np.random.rand(10)
    #plot_random(x, y, 'test.png')
    plot_test_dec(x, y, 'test.png')
    retrieve_plot('test.png')
