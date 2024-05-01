import re
from typing import Dict, List, Tuple

import fire
from datasets import Dataset
from github import Github
from huggingface_hub import create_repo

from config import ACCESS_TOKEN, HF_USERNAME

QUESTION_LIMIT = 300
CONTEXT_LIMIT = 500
ANSWER_LIMIT = 3000


def get_indent_level(line: str) -> int:
    return int((len(line) - len(line.lstrip())) / 4)


def is_function(line: str) -> bool:
    # be careful of keyword inside of strings
    multiline = re.findall(r'""".*(@).*"""', line)
    quote = re.findall(r'".*(@).*"', line)
    comment = re.findall(r"#.*(@)", line)

    if multiline:
        line = line.replace(multiline[0], "")

    if quote:
        line = line.replace(quote[0], "")

    if comment:
        line = line.replace(comment[0], "")

    multiline = re.findall(r'""".*(def).*"""', line)
    quote = re.findall(r'".*(def).*"', line)
    comment = re.findall(r"#.*(def)", line)

    if multiline:
        line = line.replace(multiline[0], "")

    if quote:
        line = line.replace(quote[0], "")

    if comment:
        line = line.replace(comment[0], "")

    return line.startswith("@") or "def" in line


def is_class(line: str) -> bool:
    # be careful of keyword inside of strings
    multiline = re.findall(r'""".*(class).*"""', line)
    quote = re.findall(r'".*(class).*"', line)
    comment = re.findall(r"#.*(class)", line)

    if multiline:
        line = line.replace(multiline[0], "")

    if quote:
        line = line.replace(quote[0], "")

    if comment:
        line = line.replace(comment[0], "")

    return line.startswith("class")


def valid_line(line: str) -> bool:
    return line.strip() != ""


def is_comment(line: str) -> bool:
    return line.strip().startswith("#") or '"""' in line


def get_python_text_files(repo, ignore_files=["test"]) -> Dict[str, str]:
    text_files = {}
    contents = repo.get_contents("")
    while contents:
        content = contents.pop(0)

        if content.path in ignore_files:
            continue

        if content.type == "dir":
            contents.extend(repo.get_contents(content.path))
            continue

        if not content.path.endswith(".py"):
            continue

        try:
            text_files[content.path] = content.decoded_content.decode("utf-8")
        except UnicodeDecodeError:
            print(f"skipping file {content.path} due to decoding error 🐝")

    return text_files


def compress_decorators(lines) -> List[Tuple[int, str]]:
    res, prefix = [], ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@"):
            prefix = stripped + " "
            continue

        level = get_indent_level(line)

        res.append((level, prefix + stripped))
        prefix = ""
    return res


def compress_comments(pairs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    res = []
    buffer = ""
    indent_start = 0
    in_comment = False
    for indent, line in pairs:
        if '"""' in line:
            if in_comment:
                res.append((indent_start, buffer + line))
                in_comment = False
                continue
            else:
                indent_start = indent
                in_comment = True
                buffer = line
                continue
        if in_comment:
            buffer += line
        else:
            res.append((indent, line))
    return res


def remove_trailing_comments(pairs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    res = []
    for indent, line in pairs:
        if indent == 0 and is_comment(line):
            continue

        # be careful of # inside of strings
        multiline = re.findall(r'""".*""".*(#.*)', line)
        quote = re.findall(r'".*".*(#.*)', line)

        if multiline:
            line = line.replace(multiline[0], "")

        if quote:
            line = line.replace(quote[0], "")

        res.append((indent, line))

    return res


def remove_imports(pairs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    res = []
    for indent, line in pairs:
        if indent == 0 and ("import " in line or "from " in line):
            continue
        res.append((indent, line))
    return res


def compress_functions(pairs: List[Tuple[int, str]]):
    res = []
    stack = [(0, res)]

    for indent, line in pairs:
        while stack and indent <= stack[-1][0]:
            stack.pop()

        if is_function(line) or is_class(line):
            new_list = [line]
            if stack:
                stack[-1][1].append(new_list)
            else:
                res.append(new_list)
            stack.append((indent, new_list))
        else:
            if stack:
                stack[-1][1].append(line)
            else:
                res.append(line)
    return res


def get_type(entry) -> Tuple[str, str]:
    return (
        ("function", entry[0].split("(")[0].split("def ")[-1])
        if is_function(entry[0])
        else ("class", entry[0].split("(")[0].split("class ")[-1])
    )


def create_entry(question: str, context: str, answer: str):
    entry = {
        "question": question.strip(),
        "context": context.strip(),
        "answer": answer.strip(),
    }
    return entry


def flatten(lst):
    if not lst:
        return [], []
    if isinstance(lst[0], list):
        answer_list, comments_list = flatten(lst[0])
        answer_rest, comments_rest = flatten(lst[1:])
        return answer_list + answer_rest, comments_list + comments_rest
    if is_comment(lst[0]):
        answer_rest, comments_rest = flatten(lst[1:])
        return answer_rest, [lst[0]] + comments_rest
    else:
        answer_rest, comments_rest = flatten(lst[1:])
        return [lst[0]] + answer_rest, comments_rest


def create_context(*argv):
    return " ".join([context.strip() for context in argv])


def create_entries(lst, file_name: str) -> List[Dict[str, str]]:
    entries = []
    context1 = ""

    for line1 in lst:
        # outer level class or function
        if isinstance(line1, list):
            type2, name2 = get_type(line1)
            context2 = ""
            answer2 = ""

            # iterate through outer level class or function contents
            for line2 in line1:
                # if embedded class or function
                if isinstance(line2, list):
                    type3, name3 = get_type(line2)
                    context3 = ""
                    answer3 = ""

                    for line3 in line2:
                        if isinstance(line3, list):
                            type4, name4 = get_type(line3)
                            context4 = ""
                            answer4 = ""

                            for line4 in line3:
                                if isinstance(line4, list):
                                    type5, name5 = get_type(line4)
                                    context5 = ""
                                    answer5 = ""

                                    for line5 in line4:
                                        if isinstance(line5, list):
                                            type6, name6 = get_type(line5)
                                            answer6, context6 = flatten(line5[1:])
                                            answer6, context6 = (
                                                " ".join(answer6),
                                                " ".join(context6),
                                            )
                                            entry6 = create_entry(
                                                question=f"In file {file_name}, create a {type6[0]} with declaration: {line5[0][:-1]}. This is embedded in a {type5} with name {name5}.",
                                                context=f"{context6}.",
                                                answer=f"{answer6}.",
                                            )
                                            entries.append(entry6)
                                            answer5 += type5 + name5 + " "

                                        elif is_comment(line5):
                                            context5 += line5 + " "
                                        else:
                                            answer5 += line5 + " "

                                    _context5 = create_context(
                                        context1, context2, context3, context4, context5
                                    )
                                    answer5 = answer5.strip()
                                    entry5 = create_entry(
                                        question=f"In file {file_name}, create a {type5[0]} with declaration: {line4[0][:-1]}. This is embedded in a {type4} with name {name4}.",
                                        context=f"{_context5}.",
                                        answer=f"{answer5}.",
                                    )
                                    entries.append(entry5)
                                    answer4 += type4 + name4 + " "

                                elif is_comment(line4):
                                    context4 += line4 + " "
                                else:
                                    answer4 += line4 + " "

                            _context4 = create_context(
                                context1, context2, context3, context4
                            )
                            answer4 = answer4.strip()
                            entry4 = create_entry(
                                question=f"In file {file_name}, create a {type4[0]} with declaration: {line3[0][:-1]}. This is embedded in a {type3} with name {name3}",
                                context=f"{_context4}.",
                                answer=f"{answer4}.",
                            )
                            entries.append(entry4)
                            answer3 += type3 + name3 + " "

                        elif is_comment(line3):
                            context3 += line3 + " "
                        else:
                            answer3 += line3 + " "

                    _context3 = create_context(context1, context2, context3)
                    answer3 = answer3.strip()

                    entry3 = create_entry(
                        question=f"In file {file_name}, create a {type3[0]} with declaration: {line2[0][:-1]}. This is embedded in a {type2} with name {name2}.",
                        context=f"{_context3}.",
                        answer=f"{answer3}.",
                    )
                    entries.append(entry3)
                    answer2 += type2 + name2 + " "

                elif is_comment(line2):
                    context2 += line2 + " "
                else:
                    answer2 += line2 + " "

            _context2 = create_context(context1, context2)
            answer2 = answer2.strip()

            entry2 = create_entry(
                question=f"In file {file_name}, create a {type2[0]} with declaration: {line1[0][:-1]}.",
                context=f"{_context2}.",
                answer=f"{answer2}.",
            )
            entries.append(entry2)

        else:
            context1 += line1 + " "
    return entries


def split_string(string: str, n: int):
    size = len(string) // n
    if n == 2:
        return string[:size], string[size:]
    return string[:size], string[size : 2 * size], string[2 * size :]


def clean_entries(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned_entries = []
    for entry in entries:
        if (
            len(entry["question"]) > QUESTION_LIMIT
            or len(entry["answer"]) > ANSWER_LIMIT
        ):
            continue

        if len(entry["context"]) > CONTEXT_LIMIT:
            context = entry["context"][:CONTEXT_LIMIT]
        else:
            context = entry["context"]

        if len(entry["answer"]) <= 1000:
            cleaned_entry = {
                "question": entry["question"],
                "context": context,
                "answer": entry["answer"],
            }
            cleaned_entries.append(cleaned_entry)
        elif len(entry["answer"]) > 1000 and len(entry["answer"]) <= 2000:
            first, second = split_string(entry["answer"], 2)
            cleaned_entry = {
                "question": entry["question"][:-1] + "(first half).",
                "context": context,
                "answer": first,
            }
            cleaned_entries.append(cleaned_entry)
            cleaned_entry = {
                "question": entry["question"][:-1] + "(second half).",
                "context": context,
                "answer": second,
            }
            cleaned_entries.append(cleaned_entry)
        else:
            first, second, third = split_string(entry["answer"], 3)
            cleaned_entry = {
                "question": entry["question"][:-1] + "(first portion).",
                "context": context,
                "answer": first,
            }
            cleaned_entries.append(cleaned_entry)
            cleaned_entry = {
                "question": entry["question"][:-1] + "(second portion).",
                "context": context,
                "answer": second,
            }
            cleaned_entries.append(cleaned_entry)
            cleaned_entry = {
                "question": entry["question"][:-1] + "(third portion).",
                "context": context,
                "answer": third,
            }
            cleaned_entries.append(cleaned_entry)

    return cleaned_entries


def extract_repo(repo_owner: str, repo_name: str) -> None:
    g = Github(ACCESS_TOKEN)
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    print(f"extracting from {repo_owner}/{repo_name} ⛏️")

    python_text_files = get_python_text_files(repo)

    entries = []

    for file_name, text in python_text_files.items():
        lines = [line for line in text.split("\n") if valid_line(line)]
        pairs = compress_comments(compress_decorators(lines))
        pairs = remove_imports(remove_trailing_comments(pairs))
        pairs = compress_functions(pairs)
        entries.extend(create_entries(pairs, file_name))

    entries = clean_entries(entries)
    dataset = Dataset.from_list(entries)
    print(f"pushing {repo_name} to hub 🚀")
    dataset.push_to_hub(repo_name)


def main(repo_url: str, create: bool = False):
    repo_owner, repo_name = repo_url.split("/")[-2:]
    repo_id = f"{HF_USERNAME}/{repo_name}"
    if create:
        create_repo(repo_id, repo_type="dataset", private=False)
    extract_repo(repo_owner=repo_owner, repo_name=repo_name)


if __name__ == "__main__":
    fire.Fire(main)
