import weave
weave.init("uuid")

@weave.op()
def simple_operation(input_value):
    return f"Processed {input_value}"

# Execute the operation and retrieve the result and call ID
result, call = simple_operation.call("example input")
call_id = call.id
print(call_id)