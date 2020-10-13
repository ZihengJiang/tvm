import re
import subprocess
from collections import namedtuple
from .expr import *
from .registry import lookup

class ExprFunctor:
    def __init__(self):
        self.memo_map = {}

    def visit(self, expr):
        if expr in self.memo_map:
            return self.memo_map[expr]

        if isinstance(expr, ObjectDef):
            res = self.visit_object_def(expr)
        elif isinstance(expr, ObjectRefDef):
            res = self.visit_object_ref_def(expr)
        elif isinstance(expr, FieldDef):
            res = self.visit_field_def(expr)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        self.memo_map[expr] = res
        return res

    def visit_object_def(self, _):
        raise NotImplementedError()

    def visit_object_ref_def(self, _):
        raise NotImplementedError()

    def visit_field_def(self, _):
        raise NotImplementedError()

class AttrsCollector(ExprFunctor):
    def __init__(self):
        super(AttrsCollector, self).__init__()
        self.fields = []

    def visit_object_def(self, obj):
        if obj.base is not None:
            self.visit(obj.base)
        if len(obj.fields) > 0:
            self.fields += obj.fields


class CodeGenCPP(ExprFunctor):
    def generate_class_comment(self, obj):
        root = obj.comment['Root']
        print(obj.comment.keys())
        root[0] = '\\brief ' + root[0].strip()
        ret = ['/*!']
        for line in root:
            ret.append(' * ' + line.strip())

        # for see also
        if 'See Also' in obj.comment:
            ret.append(' * ')
            sa = obj.comment['See Also']
            sa[0] = '\\sa ' + sa[0].strip()
            for line in sa:
                ret.append(' * ' + line.strip())

        ret.append(' */\n')
        return '\n'.join(ret)

    def generate_fields_comment(self, obj):
        if 'Attributes' not in obj.comment:
            return ''
        field_keys = [field.name for field in obj.fields]
        lines = obj.comment['Attributes']
        delimiter = []
        for lnum, line in enumerate(lines):
            if line.strip() in field_keys:
                delimiter.append(lnum)
        delimiter.append(len(lines))
        ret = {}
        for idx in range(len(delimiter) - 1):
            key = lines[delimiter[idx]].strip()
            begin_lnum = delimiter[idx] + 1
            end_lnum = delimiter[idx + 1]
            if begin_lnum + 1 == end_lnum:
                # single line comment 
                ret[key] = '/*! \\brief ' + lines[begin_lnum].strip() + ' */'
            else:
                # multiple lines comment 
                ret_lines = ['/*!']
                ret_lines.append('   * \\brief ' + lines[begin_lnum].strip())
                for line in lines[begin_lnum+1:end_lnum]:
                    ret_lines.append('   * ' + line.strip())
                ret_lines.append('   */')
                ret[key] = '\n'.join(ret_lines)
        return ret

    def generate_fvisit_attrs(self, obj, fields):
        if not obj.fvisit_attrs or len(fields) == 0:
            return ""
        template = \
        "\n" \
        "  void VisitAttrs(AttrVisitor* v) {{"\
        "{fields_str}" \
        "\n  }}"
        fields_str = [""]
        for field in fields:
            line = "v->Visit(\"{field_name}\", &{field_name});"
            fields_str.append(line.format(field_name=field.name))
        fields_str = '\n    '.join(fields_str)
        return template.format(fields_str=fields_str)

    def generate_fsequal_reduce(self, obj, fields):
        if not obj.fsequal_reduce:
            return ""
        template = \
        "\n" \
        "  void SEqualReduce(const {obj_name}* other, SEqualReducer equal) const {{\n" \
        "    return {fields_str}\n" \
        "  }}"
        fields_str = []
        for field in fields:
            line = "equal({field_name}, other->{field_name})"
            fields_str.append(line.format(field_name=field.name))
        fields_str = ' && '.join(fields_str)
        return template.format(obj_name=obj.name, fields_str=fields_str)

    def generate_fshash_reduce(self, obj, fields):
        if not obj.fshash_reduce:
            return ""
        template = \
        "\n" \
        "  void SHashReduce(SHashReducer hash_reducer) const {{\n" \
        "    {fields_str}\n" \
        "  }}"
        fields_str = []
        for field in fields:
            line = "hash_reducer({field_name});"
            fields_str.append(line.format(field_name=field.name))
        fields_str = '\n    '.join(fields_str)
        return template.format(fields_str=fields_str)

    def visit_object_def(self, obj):
        template = \
        "{comment}" \
        "class {name} : public {base_name} {{\n" \
        " public:" \
        "{fields}" \
        "{fvisit_fields}" \
        "{fsequal_reduce}" \
        "{fshash_reduce}" \
        "\n" \
        "  static constexpr const char* _type_key = \"{type_key}\";\n" \
        "  TVM_DECLARE_BASE_OBJECT_INFO({name}, {base_name});\n" \
        "}};"
        comment = self.generate_class_comment(obj)
        fields_comment = self.generate_fields_comment(obj) 
        base_name = obj.base.name
        fields = [""]
        for field in obj.fields:
            fields.append(fields_comment[field.name])
            fields.append(self.visit(field))
        fields = "\n  ".join(fields)

        collector = AttrsCollector()
        collector.visit(obj)
        fvisit_fields = self.generate_fvisit_attrs(obj, collector.fields)
        fsequal_reduce = self.generate_fsequal_reduce(obj, collector.fields)
        fshash_reduce = self.generate_fshash_reduce(obj, collector.fields)

        src = template.format(comment=comment,
                              name=obj.name,
                              base_name=base_name,
                              fields=fields,
                              fvisit_fields=fvisit_fields,
                              fsequal_reduce=fsequal_reduce,
                              fshash_reduce=fshash_reduce,
                              type_key=obj.type_key)
        return src

    def generate_object_ref_comment(self, objref):
        obj_name = objref.internal.name
        ret = []
        ret.append('/*!')
        ret.append(' * \\brief Managed reference to {}'.format(obj_name))
        ret.append(' *')
        ret.append(' * \\sa {}'.format(obj_name))
        ret.append(' */\n')
        return '\n'.join(ret)

    def visit_object_ref_def(self, objref):
        template = \
        "{comment}" \
        "class {name} : public {base_name} {{\n" \
        " public:\n" \
        "  TVM_DEFINE_OBJECT_REF_METHODS({name}, {base_name}, {obj_name});\n" \
        "}};"
        comment = self.generate_object_ref_comment(objref)
        base_name = objref.base.name
        obj_name = objref.internal.name
        src = template.format(comment=comment,
                              name=objref.name,
                              base_name=base_name,
                              obj_name=obj_name)
        return src

    def visit_field_def(self, field):
        src = "{type_name} {name};".format(type_name=field.type_.name,
                                           name=field.name) 
        return src


def generate(expr, language='cpp'):
    if language == 'cpp':
        return CodeGenCPP().visit(expr)


TextGroup = namedtuple("TextGroup", ['lines', 'is_normal'])
InternalUsed = ['end', 'custom-begin', 'custom-end']

def _process_schema_group(group):
    lines = group.lines
    match = re.match(r"//[\s]*tschema[\s]*:[\s]*([\w]+)", lines[0])
    key = match[1]
    print(key)
    expr = lookup(key)
    content = generate(expr)
    content = content.split('\n')

    # find customized content
    custom_begin = []
    custom_end = []
    for num, line in enumerate(lines):
        if re.search(r"//[\s]*tschema[\s]*:[\s]*custom-begin", line):
            custom_begin.append(num)
        if re.search(r"//[\s]*tschema[\s]*:[\s]*custom-end", line):
            custom_end.append(num)

    assert(len(custom_begin) == len(custom_end))
    num_pairs = len(custom_begin)
    custom_groups = []
    for begin, end in zip(custom_begin, custom_end):
        custom_groups.append(lines[begin:end+1])

    code = []
    code.append(lines[0])
    code += content[:-1]
    for group in custom_groups:
        code += group
    code.append(content[-1])
    code.append(lines[-1])
    return code


def process(fname, out_fname=None):
    with open(fname, 'r') as fin: 
        text = fin.read()
    lines = text.split('\n')
    begin_num = []
    end_num = []
    for num, line in enumerate(lines):
        if re.search(r"//[\s]*tschema[\s]*:[\s]*[-\w]+", line):
            match = re.findall(r"//[\s]*tschema[\s]*:[\s]*([-\w]+)", line)
            key = match[0]
            if key not in InternalUsed:
                begin_num.append(num)
        if re.search(r"//[\s]*tschema[\s]*:[\s]*end", line):
            end_num.append(num)

    assert(len(begin_num) == len(end_num))
    num_pairs = len(begin_num)
    if num_pairs == 0:
        return 

    # separate lines into groups
    groups = []
    idx = 0
    pair_idx = 0
    while idx < len(lines) and pair_idx < num_pairs:
        begin_idx = begin_num[pair_idx]
        end_idx = end_num[pair_idx]
        groups.append(TextGroup(lines[idx: begin_idx], is_normal=True))
        groups.append(TextGroup(lines[begin_idx: end_idx+1], is_normal=False))
        pair_idx += 1
        idx = end_idx + 1
    if idx < len(lines):
        groups.append(TextGroup(lines[idx:], is_normal=True))

    new_lines = []
    for group in groups:
        if group.is_normal:
            new_lines += group.lines
        else:
            new_lines += _process_schema_group(group)

    if out_fname is None:
        out_fname = fname
    with open(out_fname, 'w+') as fout:
        fout.write('\n'.join(new_lines))

    # format
    subprocess.call(["clang-format", "-assume-filename=cpp", "-i", out_fname])
    return
