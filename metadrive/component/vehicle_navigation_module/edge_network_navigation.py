from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation


class EdgeNetworkNavigation(BaseNavigation):
    """
   This class define a helper for localizing vehicles and retrieving navigation information.
   It now only support EdgeRoadNetwork
   """

    def __init__(
            self,
            engine,
            show_navi_mark: bool = False,
            random_navi_mark_color=False,
            show_dest_mark=False,
            show_line_to_dest=False
    ):
        super(EdgeNetworkNavigation, self).__init__(engine, show_navi_mark, random_navi_mark_color, show_dest_mark,
                                                    show_line_to_dest)


