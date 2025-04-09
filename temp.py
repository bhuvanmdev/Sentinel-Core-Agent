def get_tree(partition: str) -> str:
    """Get all files as a tree of c or d partition and tree as a string"""
    print(f"Executing resource 'get_tree' with path={partition}")
    # try:
    import os
    tree = []
    if not os.path.exists(partition+":"):
        return f"Partition {partition} does not exist."
    for root, dirs, files in os.walk(partition+":"):
        level = root.replace(partition, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
    result = "\n".join(tree)
    print(f"Result: {result}")
    return result
    # except Exception as e:
    #     print(f"Error getting tree: {e}")
    #     return str(e)
print(get_tree("e")) # get tree with default depth parameter
# get tree with another depth parameter
