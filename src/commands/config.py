def config (setting_key, setting_value):
  if setting_value == 'true':
    setting_value = True
  elif setting_value == 'false':
    setting_value = False
  elif '.' in setting_value:
    setting_value = float(setting_value)
  elif setting_value.isdigit():
    setting_value = int(setting_value)
  else:
    setting_value = str(setting_value)
  
  with open('src/settings.py', 'r+') as settings_file:
    content = settings_file.read()
    lines = content.split('\n')
    line = [line for line in lines if line.startswith(f'{setting_key.upper()} = ')][0]
    content = content.replace(line, f'{setting_key.upper()} = {setting_value}')
    settings_file.seek(0)
    settings_file.write(content)
