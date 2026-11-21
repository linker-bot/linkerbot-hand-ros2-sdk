"""HOP profiles must survive manual and discovered ROS construction without I/O."""

from dataclasses import replace

import pytest
import rclpy
from linkerbot_hand import (
    HandSide,
    ProtocolKind,
    TelemetrySemantic,
    TransportKind,
    ValidationError,
    get_declared_capabilities,
)
from linkerbot_hand.discovery import DiscoveryAddress, DiscoveryConfig
from linkerbot_hand_ros.discovery_deployment import (
    deployment_from_selection,
    load_deployment,
    save_deployment,
    select_hands,
)
from linkerbot_hand_ros.driver_config import (
    IDENTITY_PARAMETERS,
    read_driver_configuration,
)
from linkerbot_hand_ros.node import LinkerbotHandNode
from linkerbot_hand_ros.parameter_updates import PARAMETER_DEFAULTS
from linkerbot_hand_ros.transport_factory import TransportFactories, create_hand_device
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from test_discovery_launch import _hand_launch
from test_discovery_startup import _candidate, _report
from test_transport_factory import ConstructorSpy, NoIOTransport


def parameters(**updates):
    return {**PARAMETER_DEFAULTS, **updates}


@pytest.mark.parametrize("model", ["O6", "O30"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("wire", ["can", "can_fd"])
def test_manual_hop_profile_reaches_real_sdk_and_transport_without_io(
    model, side, wire
):
    values = parameters(
        model=model,
        side=side,
        protocol="hop",
        hop_transport=wire,
        hop_request_id=0x123,
        force_fn_hz=10.0,
    )
    config = read_driver_configuration(values.__getitem__)
    transport = NoIOTransport()
    classic = ConstructorSpy(result=transport)
    fd = ConstructorSpy(result=transport)
    device = create_hand_device(
        config, transport_factories=TransportFactories(classic, fd, ConstructorSpy())
    )
    try:
        caps = device.get_capabilities()
        assert caps.protocol is ProtocolKind.HOP
        assert caps.transport.kind is TransportKind(wire)
        assert config.device_config.hop_request_id == 0x123
        assert config.declared_capabilities.telemetry_supported(TelemetrySemantic.FORCE)
        assert len(classic.calls) == (wire == "can")
        assert len(fd.calls) == (wire == "can_fd")
        assert transport.io_calls == 0
    finally:
        device.close()


@pytest.mark.parametrize(
    "model,protocol,wire",
    [
        ("O6", ProtocolKind.LEGACY, TransportKind.CAN),
        ("O30", ProtocolKind.HOP, TransportKind.CAN_FD),
    ],
)
def test_default_protocol_and_transport_remain_compatible(model, protocol, wire):
    config = read_driver_configuration(parameters(model=model).__getitem__)
    assert config.declared_capabilities.protocol is protocol
    assert config.declared_capabilities.transport.kind is wire
    assert config.device_config.hop_request_id is None


@pytest.mark.parametrize(
    "updates",
    [
        {"protocol": "invalid"},
        {"protocol": True},
        {"hop_transport": "canfd"},
        {"hop_transport": None},
        {"hop_request_id": True},
        {"hop_request_id": -2},
        {"hop_request_id": 0x3FF},
        {"hop_brs": 1},
        {"hop_transport": "can", "hop_brs": True},
        {"model": "L6", "hop_transport": "can"},
        {"protocol": "legacy", "hop_request_id": 1},
    ],
)
def test_invalid_hop_configuration_rejected_before_construction(updates):
    with pytest.raises(ValidationError):
        read_driver_configuration(
            parameters(**{"model": "O6", "protocol": "hop", **updates}).__getitem__
        )


def hop_candidate(model, wire, brs=False):
    base = _candidate(model, HandSide.RIGHT)
    caps = get_declared_capabilities(
        model, side="right", protocol="hop", transport_kind=wire
    )
    caps = replace(caps, identity=replace(caps.identity, verified=True))
    channel = replace(
        base.channel,
        probe_protocols=(wire,),
        kind=wire,
        data_bitrate=None if wire is TransportKind.CAN else 5_000_000,
    )
    return replace(
        base,
        capabilities=caps,
        channel=channel,
        wire_protocol=wire,
        hop_brs=brs,
        address=DiscoveryAddress("hop_endpoint", 0x123, 0x523, False),
        transport=replace(base.transport, kind=wire, data_bitrate=channel.data_bitrate),
    )


@pytest.mark.parametrize("model", ["O6", "O30"])
@pytest.mark.parametrize(
    "wire,brs",
    [
        (TransportKind.CAN, False),
        (TransportKind.CAN_FD, False),
        (TransportKind.CAN_FD, True),
    ],
)
def test_discovered_node_uses_full_sdk_recipe_and_locks_static_parameters(
    model, wire, brs
):
    candidate = hop_candidate(model, wire, brs)
    rclpy.init()
    node = LinkerbotHandNode(discovery_candidate=candidate)
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        config = node._configuration
        assert config.declared_capabilities is candidate.capabilities
        assert config.device_config.hop_request_id == 0x123
        assert config.device_config.hop_transport is wire
        assert config.device_config.hop_brs is brs
        assert node._device.get_capabilities().protocol is ProtocolKind.HOP
        assert node._device.get_capabilities().transport.kind is wire
        assert not node._device.get_health().connected
        result = node.set_parameters_atomically([Parameter("hop_request_id", value=1)])
        assert not result.successful and "cleanup" in result.reason
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


@pytest.mark.parametrize("model", ["O6", "O30"])
def test_discovery_selection_and_persisted_binding_validate_hop_tactile(
    model, tmp_path
):
    candidate = hop_candidate(model, TransportKind.CAN)
    selected = select_hands(
        DiscoveryConfig(),
        scan=lambda _: _report(candidate),
        emit=lambda _: None,
        parameters={"force_fn_hz": 10.0},
    )
    assert selected[0].binding.parameters["force_fn_hz"] == 10.0
    assert not IDENTITY_PARAMETERS.intersection(selected[0].binding.parameters)
    deployment = deployment_from_selection(DiscoveryConfig(), selected, True)
    path = tmp_path / "hop.yaml"
    save_deployment(path, deployment)
    loaded = load_deployment(path)
    assert loaded.hands[0].parameters["force_fn_hz"] == 10.0
    assert loaded.hands[0].match["wire_protocol"] == "can"


@pytest.mark.parametrize(
    "name,value",
    [
        ("protocol", "hop"),
        ("hop_transport", "can"),
        ("hop_request_id", "1"),
        ("hop_brs", "true"),
    ],
)
def test_discovery_launch_rejects_manual_hop_overrides(name, value):
    with pytest.raises(ValueError, match="explicit manual parameters"):
        _hand_launch({"discover": "true", name: value})
