# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: OpenBT.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='OpenBT.proto',
  package='openbt_proto1',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0cOpenBT.proto\x12\ropenbt_proto1\"v\n\x03\x66it\x12\x0e\n\x06ndpost\x18\x01 \x03(\x05\x12\r\n\x05nskip\x18\x02 \x03(\x05\x12\x0e\n\x06nadapt\x18\x03 \x03(\x05\x12\r\n\x05power\x18\x04 \x03(\x05\x12\x0c\n\x04\x62\x61se\x18\x05 \x03(\x05\x12#\n\x05model\x18\x0f \x03(\x0e\x32\x14.openbt_proto1.Model\"\x16\n\x04\x66itp\x12\x0e\n\x06ndpost\x18\x01 \x03(\x05*\x7f\n\x05Model\x12\t\n\x05\x64ummy\x10\x00\x12\x06\n\x02\x62t\x10\x01\x12\x0c\n\x08\x62inomial\x10\x02\x12\x0b\n\x07poisson\x10\x03\x12\x08\n\x04\x62\x61rt\x10\x04\x12\t\n\x05hbart\x10\x05\x12\n\n\x06probit\x10\x06\x12\x12\n\x0emodifiedprobit\x10\x07\x12\x13\n\x0fmerck_truncated\x10\x08\x62\x06proto3')
)

_MODEL = _descriptor.EnumDescriptor(
  name='Model',
  full_name='openbt_proto1.Model',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='dummy', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='bt', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='binomial', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='poisson', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='bart', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='hbart', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='probit', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='modifiedprobit', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='merck_truncated', index=8, number=8,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=175,
  serialized_end=302,
)
_sym_db.RegisterEnumDescriptor(_MODEL)

Model = enum_type_wrapper.EnumTypeWrapper(_MODEL)
dummy = 0
bt = 1
binomial = 2
poisson = 3
bart = 4
hbart = 5
probit = 6
modifiedprobit = 7
merck_truncated = 8



_FIT = _descriptor.Descriptor(
  name='fit',
  full_name='openbt_proto1.fit',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ndpost', full_name='openbt_proto1.fit.ndpost', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nskip', full_name='openbt_proto1.fit.nskip', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nadapt', full_name='openbt_proto1.fit.nadapt', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='power', full_name='openbt_proto1.fit.power', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base', full_name='openbt_proto1.fit.base', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='openbt_proto1.fit.model', index=5,
      number=15, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=149,
)


_FITP = _descriptor.Descriptor(
  name='fitp',
  full_name='openbt_proto1.fitp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ndpost', full_name='openbt_proto1.fitp.ndpost', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=173,
)

_FIT.fields_by_name['model'].enum_type = _MODEL
DESCRIPTOR.message_types_by_name['fit'] = _FIT
DESCRIPTOR.message_types_by_name['fitp'] = _FITP
DESCRIPTOR.enum_types_by_name['Model'] = _MODEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

fit = _reflection.GeneratedProtocolMessageType('fit', (_message.Message,), dict(
  DESCRIPTOR = _FIT,
  __module__ = 'OpenBT_pb2'
  # @@protoc_insertion_point(class_scope:openbt_proto1.fit)
  ))
_sym_db.RegisterMessage(fit)

fitp = _reflection.GeneratedProtocolMessageType('fitp', (_message.Message,), dict(
  DESCRIPTOR = _FITP,
  __module__ = 'OpenBT_pb2'
  # @@protoc_insertion_point(class_scope:openbt_proto1.fitp)
  ))
_sym_db.RegisterMessage(fitp)


# @@protoc_insertion_point(module_scope)
