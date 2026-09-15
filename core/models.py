class Satellite:
    def __init__(self, id: int, name: str, frequency: float, decimation_factor: int):
        self.id = id
        self.name = name
        self.frequency = frequency
        self.decimation_factor = decimation_factor

    def __repr__(self):
        return (
            f"Satellite(id={self.id}, name={self.name!r}, "
            f"frequency={self.frequency}, decimation_factor={self.decimation_factor})"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "frequency": self.frequency,
            "decimation_factor": self.decimation_factor,
        }
