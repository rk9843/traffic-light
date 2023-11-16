class ImageInfo:
    def __init__(self, boundingBox):
        self.boundingbox = boundingBox
        return
    def GetVelocity(self):
        return self.velocity
    def GetBoundingBox(self):
        return self.boundingbox
    def GetPosition(self):
        return self.position
    def GetBackground(self):
        return self.background
    def SetPosition(self,position):
        self.position = position
    def SetBoundingBox(self, boundingbox):
        self.boundingbox = boundingbox
    def SetVelocity(self, velocity):
        self.velocity = velocity
    def SetBackground(self, background):
        self.background = background